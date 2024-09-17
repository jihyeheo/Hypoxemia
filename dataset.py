from tqdm import tqdm
import numpy as np
from tensorflow import keras
from collections import Counter
import math
import pickle
import copy

class WaveformsDataLoader(keras.utils.Sequence):
    def __init__(self, data_type, file_list, cfg):
        super(WaveformsDataLoader, self).__init__()
        self.type = data_type
        self.file_list = file_list

        self.batch_size = cfg.param.batch_size
        self.sampling_rate = cfg.param.sampling_rate
        self.learning_win = cfg.param.learning_win * self.sampling_rate
        self.pred_win = cfg.param.pred_win * self.sampling_rate
        self.page_len = [0]
        idx_list = []
        self.y_list = []
        self.y_0 = 0
        self.y_1 = 0
        self.cur_page = 0

        cur = 0
        for file_path in tqdm(self.file_list):

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            label = data["label"]
            single = np.kron((data["single"][:, 0] < 95), np.ones(self.sampling_rate)).astype(np.int32)

            if len(label) >= self.learning_win + self.pred_win:
                idxs, y_valid = self._search_index(single, label)
                y_count = Counter(y_valid)
                self.y_0 += y_count[0]
                self.y_1 += y_count[1]
                idx_list.append(idxs + cur)
                self.y_list.append(y_valid)

            cur += len(label)
            self.page_len.append(cur)

        idx_list = np.concatenate(idx_list)
        self.y_list = np.concatenate(self.y_list)
        self.y0_index = idx_list[self.y_list == 0]
        self.y1_index = idx_list[self.y_list == 1]
        self.len = len(self.y1_index) * 2

        # Run all segments
        if self.type == "test":
            self.index_sampled = idx_list
            self.len = len(self.index_sampled)

    def _search_index(self, single, label, stride=200):
        # y : 1d array
        y_learn = copy.deepcopy(single[
            np.arange(self.learning_win)
            + np.arange(
                0, len(label) - self.learning_win - self.pred_win - stride + 1, stride
            )[:, None]
        ])  # n' x learning_win
        y_pred = copy.deepcopy(label[
            np.arange(self.pred_win)
            + np.arange(self.learning_win, len(label) - self.pred_win - stride + 1, stride)[
                :, None
            ]
        ])  # n' x pred_win

        y_learn = np.any(y_learn == 1, axis=1)  # n'
        y_pred = np.any(y_pred == 1, axis=1)  # n'

        valid = np.logical_not(y_learn)  # n'
        idx_list = (
            np.where(valid == True)[0] * stride
        )  # Ensure it's a 1D array of indices
        y_pred_valid = y_pred[valid]

        return idx_list, y_pred_valid

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __getitem__(self, index):
        if index == 0:
            self._reset()

        wave, single, meta, label = [], [], [], []

        batch_idx = self.index_sampled[
            index
            * self.batch_size : min(
                (index + 1) * self.batch_size, len(self.index_sampled)
            )
        ]
        if self.type == "train":
            batch_idx += np.random.randint(200, size=len(batch_idx))

        end_idx = batch_idx[-1] + self.learning_win + self.pred_win
        cur_page_idx = self.page_len[self.cur_page]

        while True:
            if self.page_len[self.cur_page] > end_idx:
                self.cur_page -= 1
                break
            if self.page_len[self.cur_page] == end_idx:
                break

            # Find the correct file and batch
            with open(self.file_list[self.cur_page], "rb") as f:
                data = pickle.load(f)

            wave.append(data["wave"])
            single.append(data["single"])
            meta.append(
                np.tile(
                    data["meta"][None, :], (data["wave"].shape[0], 1)
                )
            )  # ts_len x channel
            label.append(data["label"])
            self.cur_page += 1

            if np.isnan(data["wave"]).any() or np.isnan(data["single"]).any() or np.isnan(data["meta"]).any() or np.isnan(np.array(data["label"])).any():
                print(self.file_list[self.cur_page])

        wave = np.concatenate(wave, axis=0)
        meta = np.concatenate(meta, axis=0)
        single = np.concatenate(single, axis=0)
        label = np.concatenate(label, axis=0)

        wave = wave[
            np.arange(self.learning_win) + batch_idx[:, None] - cur_page_idx
        ]  # batch_size x win_len x channel

        meta = meta[batch_idx - cur_page_idx]  # batch_size x channel

        y_all = single[np.arange(self.learning_win // self.sampling_rate, (self.learning_win + self.pred_win) // self.sampling_rate) + (batch_idx[:, None] - cur_page_idx) // self.sampling_rate]
        y_all = np.any(y_all[:, :, 0] < 95, axis=1).astype(np.int32)  # batch_size
        single = single[np.arange(self.learning_win // self.sampling_rate) + (batch_idx[:, None] - cur_page_idx) // self.sampling_rate]
        label = label[np.arange(self.learning_win, self.learning_win + self.pred_win)+ batch_idx[:, None]- cur_page_idx]
        y_pred = np.any(label == 1, axis=1).astype(np.float32)  # batch_size

        if self.type == "train":
            return (wave, single, meta), y_all
        else:
            return (wave, single, meta), y_pred, y_all

    def _reset(self):
        self.cur_page = 0
        if not self.type == "test":
            self.index_sampled = np.sort(
                np.append(
                    np.random.choice(self.y0_index, len(self.y1_index)), self.y1_index
                )
            )
