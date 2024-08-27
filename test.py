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
from utils import analyze, FBetaScore
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

learning_win = cfg.param.learning_win
pred_win = cfg.param.pred_win
data_dir = cfg.param.data_dir
n_cv_split = cfg.param.n_cv_split
batch_size = cfg.param.batch_size
sampling_rate = cfg.param.sampling_rate
learning_rate = cfg.param.init_lr

HP_TEST_TYPE = hp.HParam("test_type")#, hp.Discrete(["int", "ext"]))
HP_LEARNING_WIN = hp.HParam("learning_win")#, hp.Discrete([60, 120, 180, 300]))
HP_PRED_WIN = hp.HParam("pred_win")#, hp.Discrete([60, 120, 180, 300]))
HP_SAMPLING_RATE = hp.HParam("sampling_rate")#, hp.Discrete([100, 200]))
HP_BATCH_SIZE = hp.HParam("batch_size")#, hp.Discrete([32, 64, 128, 512]))
HP_LEARNING_RATE = hp.HParam("learning_rate")#, hp.Discrete([1.0e-3, 1.0e-4, 1.0e-5]))


with tf.summary.create_file_writer(save_path).as_default():
	hp.hparams_config(
		hparams=[HP_TEST_TYPE, HP_LEARNING_WIN, HP_PRED_WIN, HP_SAMPLING_RATE, HP_BATCH_SIZE, HP_LEARNING_RATE],
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
int_test_path_list = sorted(glob.glob(data_dir.replace("train", "test")))
ext_test_path_list = sorted(glob.glob(data_dir.replace("SNUH", "CNUH").replace("train", "test")))
print("int test:, ", len(int_test_path_list), "ext test:", len(ext_test_path_list))
int_test_dl = WaveformsDataLoader(data_type="test", 
								  data_path=int_test_path_list, 
								  )
ext_test_dl = WaveformsDataLoader(data_type="test", 
								  data_path=ext_test_path_list
								  )


# test results
for test_type in ["int", "ext"] :
	res = []
	if n_cv_split != 0 :
		print()
		#for fold in range(1, n_cv_split+1, 1):
	else :
		test_dl = globals()[f"{test_type}_test_dl"]
		for x, y, y_all in test_dl :  
			break
		model = get_model(args.model, input_shape=x.shape[1:], n_classes=1, y_train=y)
		if args.model == "gbm":
			model.load_model(save_path + f"/model")
			y_proba = model.predict_proba(x.reshape(-1, x.shape[1]*x.shape[2]))[:, 1]

		else:
			custom_obj = {
				'FBetaScore': FBetaScore(beta=2)
			}   
			model = load_model(save_path + f"/model.keras", custom_objects=custom_obj) 

			y_list = []
			y_all_list = []
			y_proba_list = []

			for x, y, y_all in tqdm(test_dl) :
				
				y_proba = model.predict(x, verbose=False)
				y_list.append(y)
				y_all_list.append(y_all)
				y_proba_list.append(y_proba)

			y_np = np.concatenate(y_list)
			y_all_np = np.concatenate(y_all_list)
			y_proba_np = np.concatenate(y_proba_list)

			res.append((y_np, y_proba_np, y_all_np))
			params = [test_type, learning_win, pred_win, sampling_rate, batch_size, learning_rate]
			analyze(save_path, model, test_type, res, params)  # visualzation auprc, auroc
