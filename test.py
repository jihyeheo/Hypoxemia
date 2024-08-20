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
from tensorboard.plugins.hparams import api as hp

# custom import
from model import get_model
from utils import analyze, FBetaScore
from dataset import dataset


# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# args
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gbm", choices=["gbm", "lstm", "resnet", "inception", "transformer"])
parser.add_argument("--weight")
args = parser.parse_args()
save_path = f"{args.weight}" 
HP_TEST_TYPE = hp.HParam("test_type", hp.Discrete(["int", "ext"]))

# parameter
with open(save_path + "/config.yaml", "r", encoding="utf-8-sig") as f:
    cfg = edict(yaml.safe_load(f))

learning_win = cfg.train_param.learning_win
pred_win = cfg.train_param.pred_win
data_dir = "./data/processed/train/SNUH/*.npy"
n_cv_split = cfg.train_param.n_cv_split


with tf.summary.create_file_writer(save_path).as_default():
    hp.hparams_config(
    hparams=[HP_TEST_TYPE],
        metrics=[
            hp.Metric('Specificity', display_name='Specificity'),
            hp.Metric('auc', display_name='AUC'),
            hp.Metric('prc', display_name='PRC'),
            hp.Metric('f1score', display_name='F1 Score'),
            hp.Metric('f2score', display_name='F2 Score'),

            hp.Metric('Accuracy', display_name='Accuracy'),
            hp.Metric('Precision', display_name='Precision'),
            hp.Metric('Recall(PPV)', display_name='Recall(PPV)'),
            hp.Metric('NPV', display_name='NPV'),
        ]
    )

# Variables npy path list
int_test_path_list = sorted(glob.glob(data_dir.replace("train", "test")))
ext_test_path_list = glob.glob(data_dir.replace("SNUH", "CNUH").replace("train", "test"))
X_int_test, y_int_test, y_int_all, _ = dataset(int_test_path_list, learning_win, pred_win)
X_ext_test, y_ext_test, y_ext_all, _ = dataset(ext_test_path_list, learning_win, pred_win)


# test results
for test_type in HP_TEST_TYPE.domain.values :
    res = []
    for fold in range(1, n_cv_split+1, 1):
        log_path = save_path + f"{fold}_model/"
        x = np.swapaxes(globals()[f"X_{test_type}_test"], 1, 2)
        
        model = get_model(args.model, input_shape=X_int_test.shape[1:], n_classes=1, y_train=y_int_all)
        x = np.swapaxes(globals()[f"X_{test_type}_test"], 1, 2)
        y = globals()[f"y_{test_type}_test"]
        y_all = globals()[f"y_{test_type}_all"]

        if args.model == "gbm":
            model.load_model(log_path + f"{fold}_model")
            y_proba = model.predict_proba(x.reshape(-1, x.shape[1]*x.shape[2]))[:, 1]

        else:
            custom_obj = {
            'FBetaScore': FBetaScore(beta=2)
        }
            model = load_model(log_path + f"{fold}_model.keras", custom_objects=custom_obj) 
            y_proba = model.predict(x)

        res.append((y, y_proba, y_all))
    analyze(save_path, args.model, test_type, res, [learning_win, pred_win])  # visualzation auprc, auroc
