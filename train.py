import numpy as np
import argparse
import random
from datetime import datetime
from pathlib import Path
import yaml
import glob
from easydict import EasyDict as edict
import warnings
import shutil

warnings.filterwarnings("ignore")

# model import
import tensorflow as tf
from tensorflow import keras
from keras import optimizers, losses, callbacks

# custom import
from model import get_model
from utils import FBetaScore
from dataset import WaveformsDataLoader

# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Parameter
with open("config.yaml", "r", encoding="utf-8-sig") as f:
	cfg = edict(yaml.safe_load(f))

num_epochs = cfg.param.num_epochs
batch_size = cfg.param.batch_size
init_lr = cfg.param.init_lr
learning_win = cfg.param.learning_win
pred_win = cfg.param.pred_win
sampling_rate = cfg.param.sampling_rate

# Variables npy path list
path_list = sorted(glob.glob("./data/processed/train/SNUH/*.pkl"))
train_list = path_list[:int(len(path_list) * 0.8)]
validation_list = path_list[int(len(path_list) * 0.8):]

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet", choices=["lstm", "resnet", "inception"])
args = parser.parse_args()
with open("config.yaml", "r", encoding="utf-8-sig") as f:
	cfg = edict(yaml.safe_load(f))
	print(cfg)

# Load dataset and print shape
train_dataset = WaveformsDataLoader("train", train_list, cfg)
val_dataset = WaveformsDataLoader("train", validation_list, cfg)
(X_wave, X_single, X_meta), _ = train_dataset[0]
print("X_wave shape: ", X_wave.shape)
print("X_single shape: ", X_single.shape)
print("X_meta shape: ", X_meta.shape)

save_path = Path("log/" + args.model + "/" + datetime.now().strftime("%m%d_%H%M%S"))
save_path.mkdir(parents=True, exist_ok=True)

# Save config
shutil.copy("config.yaml", save_path / "config.yaml")

model = get_model(args.model, X_wave.shape[1:], use_single=True, use_meta=True)

# optimizer, loss, metrics, compile
optimizer = optimizers.Adam(learning_rate=init_lr)
loss_fn = losses.BinaryCrossentropy(name="cross_entropy")
METRICS = [
	keras.metrics.AUC(name="auc"),
	keras.metrics.AUC(name="prc", curve="PR"),
	FBetaScore(beta=2, name="f2score")
]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=METRICS)
print(model.summary())

# callback
loss_callback = callbacks.ModelCheckpoint(
	save_path / "model.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=1)
tensorboard_callback = callbacks.TensorBoard(log_dir=save_path)

# fit
history = model.fit(
	train_dataset,
	validation_data=(val_dataset),
	batch_size=batch_size,
	epochs=num_epochs,
	callbacks=[loss_callback, tensorboard_callback],
	shuffle=False, # dataset defining
).history