import numpy as np
import tensorflow as tf
import argparse
import random
import yaml
import glob
from tqdm import tqdm
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore")

# model import
from tensorflow import keras
from keras.models import load_model
from tensorboard.plugins.hparams import api as hp

# custom import
from model import get_model
from utils import analyze
from dataset import WaveformsDataLoader

# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet")#, choices=["gbm", "lstm", "resnet", "inception", "transformer", "lstm2"])
parser.add_argument("--weight")
args = parser.parse_args()
save_path = f"{args.weight}" 

# parameter
with open(save_path + "/config.yaml", "r", encoding="utf-8-sig") as f:
	cfg = edict(yaml.safe_load(f))

num_epochs = cfg.param.num_epochs
batch_size = cfg.param.batch_size
init_lr = cfg.param.init_lr
learning_win = cfg.param.learning_win
pred_win = cfg.param.pred_win
sampling_rate = cfg.param.sampling_rate


HP_TEST_TYPE = hp.HParam("test_type")#, hp.Discrete(["int", "ext"]))
HP_LEARNING_WIN = hp.HParam("learning_win")#, hp.Discrete([60, 120, 180, 300]))
HP_PRED_WIN = hp.HParam("pred_win")#, hp.Discrete([60, 120, 180, 300]))
HP_SAMPLING_RATE = hp.HParam("sampling_rate")#, hp.Discrete([100, 200]))
HP_IDX_SAMPLING_RATE = hp.HParam("idx_sampling_rate")#, hp.Discrete([0.01, 0.001, 0.0001]))
HP_BATCH_SIZE = hp.HParam("batch_size")#, hp.Discrete([32, 64, 128, 512]))
HP_LEARNING_RATE = hp.HParam("learning_rate")#, hp.Discrete([1.0e-3, 1.0e-4, 1.0e-5]))


with tf.summary.create_file_writer(save_path).as_default():
	hp.hparams_config(
		hparams=[HP_TEST_TYPE, HP_LEARNING_WIN, HP_PRED_WIN, HP_SAMPLING_RATE, HP_IDX_SAMPLING_RATE, HP_BATCH_SIZE, HP_LEARNING_RATE],
		metrics=[
			hp.Metric('auc', display_name='AUC'),
			hp.Metric('prc', display_name='PRC'),
			hp.Metric('f1score', display_name='F1 Score'),
			hp.Metric('f2score', display_name='F2 Score'),
			hp.Metric('threshold', display_name='threshold'),

			hp.Metric('Accuracy', display_name='Accuracy'),
			hp.Metric('Precision', display_name='Precision'),
			hp.Metric('Recall(PPV)', display_name='Recall(PPV)'),
			hp.Metric('NPV', display_name='NPV'),
			hp.Metric('Specificity', display_name='Specificity'),

		]
	)

# Variables npy path list
int_test_path_list = sorted(glob.glob("./data/processed/test/SNUH/*.pkl".replace("train", "test")))
ext_test_path_list = sorted(glob.glob("./data/processed/test/CNUH/*.pkl".replace("train", "test")))
print("int test:, ", len(int_test_path_list), "ext test:", len(ext_test_path_list))
int_test_dl = WaveformsDataLoader("test", int_test_path_list, cfg)
ext_test_dl = WaveformsDataLoader("test", ext_test_path_list, cfg)
(X_wave, X_single, X_meta), _, _ = int_test_dl[0]
print("X_wave shape: ", X_wave.shape)
print("X_single shape: ", X_single.shape)
print("X_meta shape: ", X_meta.shape)

model = get_model(args.model, X_wave.shape[1:], use_single=True, use_meta=True)
model.load_weights(save_path + f"/model.weights.h5")

# test results
for test_type in ["int", "ext"] :
	test_dl = globals()[f"{test_type}_test_dl"]

	res = []
	y_list = []
	y_all_list = []
	y_proba_list = []

	for x, y, y_all in tqdm(test_dl) :

		y_proba = model.predict(x, verbose=False)[:, 0]
		y_list.append(y)
		y_all_list.append(y_all)
		y_proba_list.append(y_proba)

	y_np = np.concatenate(y_list)
	y_all_np = np.concatenate(y_all_list)
	y_proba_np = np.concatenate(y_proba_list)

	res.append((y_np, y_proba_np, y_all_np))
	params = [test_type, learning_win, pred_win, sampling_rate, batch_size, init_lr]
	analyze(save_path, model, test_type, res, params)  # visualzation auprc, auroc
