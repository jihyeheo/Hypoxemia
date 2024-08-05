
import numpy as np
import tensorflow as tf
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import shutil
#from easydict import EasyDict as edict
from sklearn.preprocessing import MinMaxScaler
import glob
from collections import Counter
from tqdm import tqdm
from tensorflow import keras
from sktime.classification.deep_learning import *
from sklearn.metrics import accuracy_score
from utils import evaluate_metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Accuracy


init_lr = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)  # init_lr)  # , global_clipnorm=0.1)


# Seed
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# # Parameters
# with open("config.yaml", "r") as f:
#     cfg = edict(yaml.safe_load(f))
# init_lr = cfg.train_param.init_lr
# num_epochs = cfg.train_param.num_epochs
# input_shape = cfg.train_param.input_shape
# batch_size = cfg.train_param.batch_size
# window_size = cfg.train_param.window_size
# sample_interval = cfg.train_param.sample_interval
# data_dir = "./Create_Data/SNUH_train/dependent/"

METRICS = [
    keras.metrics.BinaryCrossentropy(name="cross entropy"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.AUC(name="prc", curve="PR"),
]

# def 
def dataset(path_list):
    X_list = []
    label_list = []
    for file_path in tqdm(path_list) :
        data = np.load(file_path, allow_pickle=True)
        x_ = data[:, :-1]
        y_ = data[:, -1]
        
        for i in range(len(x_)//30):
            label = 1 if np.sum(y_[i+30:i+60]) >= 1 and np.sum(y_[i:i+30])==0 else 0
            x = x_[i:i+30]
            X_list.append(x)
            label_list.append(label)
    #print(label_list)
    return np.array(X_list), np.array(label_list).reshape(-1,)

# X,y list
train_path_list = sorted(glob.glob("./data/processed/SNUH/train/*.npy"))
int_test_path_list = sorted(glob.glob("./data/processed/SNUH/test/*.npy"))
ext_test_path_list = glob.glob("./data/processed/CNUH/*.npy")


    
X_train, y_train = dataset(train_path_list)
X_int_test, y_int_test = dataset(int_test_path_list)
X_ext_test, y_ext_test = dataset(ext_test_path_list)

for data in [(X_train, y_train), (X_int_test, y_int_test)]:#, [X_ext_test, y_ext_test]]) :
    X,y = data
    print(X.shape, y.shape)
    print(X.shape, Counter(list(y)))




score_list = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # model
    model = CNNClassifier()
    model = model.build_model(input_shape=(30,12), n_classes=1)#(n_epochs = 20, batch_size=128)
    model.compile(optimizer=Adam(), 
                  loss=BinaryCrossentropy(name='cross_entropy'), 
                  metrics=[Accuracy(), AUC(name='auc'), AUC(name='prc', curve='PR')])




    model.fit(X_train_fold, y_train_fold,
              validation_data = (X_val_fold, y_val_fold),
              batch_size = 128,
              epochs = 100)#,
              #class_weight={0.0 : 0.1 , 1.0 : 1})#, X_val_fold, y_val_fold)
    pred = model.predict(X_val_fold)
    print(fold) 
    print("int test", evaluate_metrics(y_int_test, model.predict(X_int_test), model.predict_proba(X_int_test)[:, 0]))
    #print(accuracy_score(y_ext_test, model.predict(X_ext_test)))
    

#     model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
#     loss_callback = ModelCheckpoint(save_path / f"{idx}_model", monitor="val_auc", save_best_only=True, mode="max", verbose=1)


#     history = model.fit(X_train_fold, y_train_fold, 
#                         validation_data=(X_val_fold, y_val_fold), 
#                         verbose=1,
#                         batch_size= batch_size,
#                         epochs=num_epochs,
#                         class_weight=[],
#                         callbacks=[loss_callback])
    
#     model.save(path + f"{max_idx}.h5")
#     with open(save_path / "history.json", "w") as f:
#         json.dump(history, f)
#     visualization(history, save_path / f"{idx}.png")
#     score_list.append(model.evaluate(X_val_fold, y_val_fold))

# max_idx = score_list.index(max(score_list))
# model.load(path +  f"{max_idx}.h5")
# model.evaluate(test_dl)