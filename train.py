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
import sklearn
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.models import load_model
from imblearn.under_sampling import RandomUnderSampler
from tensorboard.plugins.hparams import api as hp

# custom import
from model import get_model
from utils import FBetaScore
from dataset import dataset


# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# parameter
with open("config.yaml", "r", encoding="utf-8-sig") as f:
    cfg = edict(yaml.safe_load(f))

num_epochs = cfg.train_param.num_epochs
batch_size = cfg.train_param.batch_size
learning_win = cfg.train_param.learning_win
pred_win = cfg.train_param.pred_win
data_dir = "./data/processed/train/SNUH/*.npy"
init_lr = cfg.train_param.init_lr
n_cv_split = cfg.train_param.n_cv_split
METRICS = [
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),
    FBetaScore(beta=2, name="f2score")
]


# args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gbm", choices=["gbm", "lstm", "resnet", "inception", "transformer"])
args = parser.parse_args()
save_path = Path("log/" + datetime.now().strftime("%m%d%H%M%S") + args.model)
os.makedirs(save_path)
shutil.copy("config.yaml", save_path / "config.yaml")

os.makedirs(save_path / "npy/")
os.makedirs(save_path / "fig/")





# hyperparameter tuning
HP_LEARNING_WIN = hp.HParam("learning_win", hp.Discrete([30, 60, 90]))
HP_PRED_WIN = hp.HParam("pred_win", hp.Discrete([30, 60, 90]))
# save_parameter_path = str(save_path) + '/hparam_tuning'
# print(save_parameter_path)
# with tf.summary.create_file_writer(save_parameter_path).as_default():
#     hp.hparams_config(
#     hparams=[HP_LEARNING_WIN, HP_PRED_WIN],
#     metrics=[
#         hp.Metric('auc', display_name='AUC'),
#         hp.Metric('prc', display_name='PRC'),
#         hp.Metric('f2score', display_name='F2 Score')
#     ]
#   )


for learning_win in HP_LEARNING_WIN.domain.values :
    for pred_win in HP_PRED_WIN.domain.values :
        
        hparams = {
            "learning_win" : learning_win,
            "pred_win" : pred_win
        }
        # Variables npy path list
        train_path_list = sorted(glob.glob(data_dir))
        X_train, y_train, y_ = dataset(train_path_list, learning_win, pred_win)

        rus = RandomUnderSampler(random_state=0)
        X_train_shape = X_train.shape[1:]
        flattened_len = X_train_shape[0] * X_train_shape[1]
        X_train, y_train = rus.fit_resample(X_train.reshape(-1, flattened_len), y_train)
        X_train = X_train.reshape((-1,) + X_train_shape)

        # # confirm dataset
        print(X_train.shape, y_train.shape, Counter(list(y_train)))


        # train training
        skf = StratifiedKFold(n_splits=n_cv_split, shuffle=True, random_state=RANDOM_SEED)
        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
            # train, val
            X_train_fold, X_val_fold = np.swapaxes(X_train[train_index], 1, 2), np.swapaxes(X_train[val_index], 1, 2)
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
                    save_path / f"lw{learning_win},pw{pred_win}/{fold}_model.keras", monitor="val_auc", save_best_only=True, mode="max", verbose=1
                )
                
                log_dir = save_path / f"lw{learning_win},pw{pred_win}/{fold}_model/"
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                history = model.fit(
                    X_train_fold,
                    y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    batch_size=batch_size,
                    epochs=num_epochs,
                    callbacks=[loss_callback, tensorboard_callback],
                ).history