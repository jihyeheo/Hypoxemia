from tqdm import tqdm
import numpy as np


def dataset(path_list ,window_size, sample_interval):
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