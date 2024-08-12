from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def dataset(path_list, learning_win, pred_win):
    X_list = []
    label_list = []
    for file_path in tqdm(path_list) :
        data = np.load(file_path, allow_pickle=True)
        x = data[:, :-1] #  n x features
        y = data[:, -1] #  n
        
        if y.shape[0] < learning_win + pred_win:
            continue
        x = sliding_window_view(x, learning_win, axis=0)[:-pred_win] #  n' x features x learning_win
        y_learn = sliding_window_view(y, learning_win, axis=0)[:-pred_win] #  n' x pred_win
        y_pred = sliding_window_view(y, pred_win, axis=0)[learning_win:] #  n' x pred_win

        assert y_learn.shape[0] == y_pred.shape[0]

        y_learn = np.any(y_learn == 1, axis=1) #  n'
        y_pred = np.any(y_pred == 1, axis=1) #  n'

        valid = np.logical_not(np.logical_and(y_learn, y_pred)) #  n'

        x = x[valid == True] #  n'' x learning_win x features
        label = y_pred[valid == True] #  n''

        X_list.append(x)
        label_list.append(label)

    return np.concatenate(X_list, axis=0), np.concatenate(label_list).astype(np.int64)