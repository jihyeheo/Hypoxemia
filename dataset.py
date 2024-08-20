from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def dataset(path_list, learning_win, pred_win):
    X_list, y_tp_list, y_all_list, demo_list = [], [], [], []

    for file_path in tqdm(path_list) :
        # if "SNUH" in file_path : 
        #     csv_file = pd.read_csv("./data/SNUH_final_vitaldb.csv")    
        #     csv_demo = csv_file[csv_file["fileid"]==file_path.split("SNUH")[-1][1:-4]].reset_index()[["sex", "age", "weight", "height"]].to_numpy()[0,:]
        #     csv_demo[0] = 1 if csv_demo[0] == 'F' else 0
        # else :
        #     csv_file = pd.read_csv("./data/CNUH_final_vitaldb.csv")
        #     csv_demo = csv_file[csv_file["filename"]==file_path.split("/")[-1]].reset_index().iloc[0][["Sex", "나이", "몸무게", "키"]].to_numpy()[0,:]
        data = np.load(file_path, allow_pickle=True)
        x = data[:, :-1] #  n x features
        y = data[:, -1] #  n
        y_all = data[:, 0] < 95 #  n, Hypoxemia defined as SpO2 < 95%

        # Skip if the size(data) < len(learning_win) + len(pred_win)
        if y.shape[0] < learning_win + pred_win:
            continue

        x = sliding_window_view(x, learning_win, axis=0)[:-pred_win] #  n' x features x learning_win
        y_learn = sliding_window_view(y, learning_win, axis=0)[:-pred_win] #  n' x pred_win
        y_pred = sliding_window_view(y, pred_win, axis=0)[learning_win:] #  n' x pred_win
        y_all = sliding_window_view(y_all, pred_win, axis=0)[learning_win:] #  n' x pred_win
        assert y_learn.shape[0] == y_pred.shape[0]

        # Regard as a positive event if any of element == 1
        y_learn = np.any(y_learn == 1, axis=1) #  n'
        y_pred = np.any(y_pred == 1, axis=1) #  n'
        y_all = np.any(y_all == 1, axis=1) #  n'

        #
        valid = np.logical_not(np.logical_and(y_learn, y_pred)) #  n'

        x = x[valid == True] #  n'' x learning_win x features
        y_tp = y_pred[valid == True] #  n''
        y_all = y_all[valid == True] #  n''

        X_list.append(x)
        y_tp_list.append(y_tp)
        y_all_list.append(y_all)
        #demo_list.append(csv_demo)

    return np.concatenate(X_list, axis=0), np.concatenate(y_tp_list).astype(np.int64), np.concatenate(y_all_list).astype(np.int64), np.array(demo_list).astype(float)