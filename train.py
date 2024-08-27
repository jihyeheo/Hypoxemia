# import packages

import numpy as np
import tensorflow as tf
import argparse
import random
from datetime import datetime
from pathlib import Path
import yaml
import shutil
import glob
from easydict import EasyDict as edict
import warnings

warnings.filterwarnings("ignore")

# model import
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from imblearn.under_sampling import RandomUnderSampler

# custom import
from model import get_model
from utils import FBetaScore
from dataset import WaveformsDataLoader


# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#tf.random.set_seed(RANDOM_SEED)


# Parameter
with open("config.yaml", "r", encoding="utf-8-sig") as f:
    cfg = edict(yaml.safe_load(f))

num_epochs = cfg.param.num_epochs
batch_size = cfg.param.batch_size
data_dir =cfg.param.data_dir
init_lr = cfg.param.init_lr
demo_shape = cfg.param.demo_shape
n_cv_split = cfg.param.n_cv_split
learning_win = cfg.param.learning_win
pred_win = cfg.param.pred_win


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet", choices=["gbm", "lstm", "resnet", "inception", "transformer"])
args = parser.parse_args()
save_path = Path("log/" + args.model + "/" + datetime.now().strftime("%m%d_%H%M"))
save_path.mkdir(parents=True, exist_ok=True)
shutil.copy("config.yaml", save_path / "config.yaml")



# Variables npy path list
path_list = sorted(glob.glob(data_dir))
train_list = sorted(path_list[:int(len(path_list) * 0.8)])
validation_list = sorted(path_list[int(len(path_list) * 0.8):])

train_dataset = WaveformsDataLoader("train", train_list)
val_dataset = WaveformsDataLoader("train", validation_list)
X_train_fold, y_train_fold = train_dataset[0]

# Undersamping 
# rus = RandomUnderSampler(random_state=RANDOM_SEED)
# X_train_shape = X_train_fold.shape[1:]
# flattened_len = X_train_fold.shape[1] * X_train_fold.shape[2] 
# X_train_fold, y_train_fold = rus.fit_resample(X_train_fold.reshape(-1, flattened_len), y_train_fold)
# X_val_fold, y_val_fold = rus.fit_resample(X_val_fold.reshape(-1, flattened_len), y_val_fold)
# X_train_fold = X_train_fold.reshape((-1,) + X_train_shape)[:,:,:3]
# X_val_fold = X_val_fold.reshape((-1,) + X_train_shape)[:,:,:3]


if n_cv_split != 0 : 
    print()
    # 나중에 코드 다시 업로드 하기
    #cross-validation
    # skf = StratifiedKFold(n_splits=n_cv_split, shuffle=True, random_state=RANDOM_SEED)
    # for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    #     X_train_fold, X_val_fold = np.swapaxes(X_train[train_index], 1, 2), np.swapaxes(X_train[val_index], 1, 2)
    #     y_train_fold, y_val_fold = y_train[train_index, np.newaxis], y_train[val_index, np.newaxis]
else : 
    log_path = save_path #/ f"{fold}_model/"
    # model load
    model = get_model(args.model, input_shape=X_train_fold.shape[1:], n_classes=1, y_train=y_train_fold)
    if args.model == "gbm":
        model.fit(train_dataset, eval_set=[val_dataset])
        model.save_model(save_path / f"model")
    else:
        # optimizer, loss, metrics, compile
        optimizer = Adam(learning_rate=init_lr, clipnorm=0.5)
        loss_fn = BinaryCrossentropy(name="cross_entropy")
        METRICS = [
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="prc", curve="PR"),
            FBetaScore(beta=2, name="f2score")
        ]
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)
        
        # callback
        loss_callback = ModelCheckpoint(
            save_path / f"model.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1)
        tensorboard_callback = TensorBoard(log_dir=save_path)
        
        # fit
        history = model.fit(
            train_dataset,
            validation_data=(val_dataset),
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[loss_callback, tensorboard_callback],
            shuffle=False, # dataset defining
        ).history