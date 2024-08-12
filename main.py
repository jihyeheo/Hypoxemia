import os
import numpy as np
import tensorflow as tf
import argparse
import random
from datetime import datetime
from pathlib import Path
import yaml
import shutil
import glob
from collections import Counter
from tqdm import tqdm
from easydict import EasyDict as edict
import json
import warnings

warnings.filterwarnings("ignore")

# model import
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.models import load_model
from imblearn.under_sampling import RandomUnderSampler

# custom import
from model import get_model
from utils import visualization, analyze
from dataset import dataset


# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# parameter
with open("config.yaml", "r") as f:
    cfg = edict(yaml.safe_load(f))

num_epochs = cfg.train_param.num_epochs
batch_size = cfg.train_param.batch_size
learning_win = cfg.train_param.learning_win
pred_win = cfg.train_param.pred_win
data_dir = "./data/processed/train/SNUH/*.npy"
init_lr = cfg.train_param.init_lr
n_cv_split = 5
METRICS = [
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),
]


# args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gbm", choices=["gbm", "cnn", "cntc", "fcn", "resnet", "inception"])
args = parser.parse_args()
save_path = Path("weights/" + datetime.now().strftime("%m%d%H%M%S") + args.model)
os.makedirs(save_path)
shutil.copy("config.yaml", save_path / "config.yaml")

os.makedirs(save_path / "npy/")
os.makedirs(save_path / "fig/")


# Variables npy path list
train_path_list = sorted(glob.glob(data_dir))
int_test_path_list = sorted(glob.glob(data_dir.replace("train", "test")))
ext_test_path_list = glob.glob(data_dir.replace("SNUH", "CNUH").replace("train", "test"))

X_train, y_train = dataset(train_path_list, learning_win, pred_win)
X_int_test, y_int_test = dataset(int_test_path_list, learning_win, pred_win)
X_ext_test, y_ext_test = dataset(ext_test_path_list, learning_win, pred_win)


#rus = RandomUnderSampler(random_state=0)
X_train_shape = X_train.shape[1:]
flattened_len = X_train_shape[0] * X_train_shape[1]
#X_train, y_train = rus.fit_resample(X_train.reshape(-1, flattened_len), y_train)
#X_train = X_train.reshape((-1,) + X_train_shape)

# confirm dataset
for X, y in [(X_train, y_train), (X_int_test, y_int_test), (X_ext_test, y_ext_test)]:
    print(X.shape, y.shape)
    print(X.shape, Counter(list(y)))

# train training
skf = StratifiedKFold(n_splits=n_cv_split, shuffle=True, random_state=RANDOM_SEED)
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    # train, val
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index, np.newaxis], y_train[val_index, np.newaxis]

    # model load
    model = get_model(args.model, input_shape=X_train_fold.shape[1:], n_classes=1, y_train=y_train_fold)
    if args.model == "gbm":
        model.fit(X_train_fold.reshape(-1, flattened_len), y_train_fold, eval_set=[(X_val_fold.reshape(-1, flattened_len), y_val_fold)])
        model.save_model(save_path / f"{fold}_model")

    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy(name="cross_entropy")
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)
        print(model.summary())
        loss_callback = tf.keras.callbacks.ModelCheckpoint(
            save_path / f"{fold}_model.keras", monitor="val_auc", save_best_only=True, mode="max", verbose=1
        )

        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[loss_callback],
            class_weight={0.0: 1, 1.0: 100},
        ).history

        with open(save_path / "history.json", "w") as f:
            json.dump(history, f)
        visualization(history, save_path / f"fig/{fold}_model.png")


# test results

for test_type in ["int", "ext"]:
    res = []
    res_low_spo2 = []
    for fold in range(n_cv_split):
        model = get_model(args.model, input_shape=X_train_fold.shape[1:], n_classes=1, y_train=y_train_fold)
        x = globals()[f"X_{test_type}_test"]
        y = globals()[f"y_{test_type}_test"]
        if args.model == "gbm":
            model.load_model(save_path / f"{fold+1}_model")
            y_proba = model.predict_proba(x.reshape(-1, flattened_len))[:, 1]

        else:
            model = load_model(save_path / f"{fold+1}_model.keras")
            y_proba = model.predict(x)

        #hypoxemia_total = np.any(x[:, 0] < 95, axis=1)
        #y_low_spo2 = y[hypoxemia_total == True]
        #y_low_spo2_pred = y_proba[hypoxemia_total == True]

        res.append((y, y_proba))
        #res_low_spo2.append((y_low_spo2, y_low_spo2_pred))
        
    analyze(save_path, args.model, test_type, res)  # visualzation auprc, auroc
    #analyze(save_path, args.model, test_type, res_low_spo2)  # visualzation auprc, auroc
