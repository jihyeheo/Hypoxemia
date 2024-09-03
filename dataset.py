from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from keras.utils import Sequence
from tqdm import tqdm
from collections import Counter
import math
from easydict import EasyDict as edict
import pandas as pd



class WaveformsDataLoader(Sequence):
	def __init__(self, data_type, data_path, cfg):
		super(WaveformsDataLoader, self).__init__()
		self.type = data_type
		self.data_file_list = data_path

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


		demo_csv = pd.read_csv("./data/data_final_vitaldb.csv")
		cur = 0
		for file_path in tqdm(self.data_file_list):
			file_name = file_path.split("/")[-1]
			print(file_name)
			data = np.load(file_path, allow_pickle=True)
			break
			y = data[:,-1]
			if len(y) >= self.learning_win + self.pred_win: 
				idxs, y_valid = self._search_index(y)
				y_count = Counter(y_valid)
				self.y_0 += y_count[0]
				self.y_1 += y_count[1]
				idx_list.append(idxs + cur)
				self.y_list.append(y_valid)
		
			cur += len(y)
			self.page_len.append(cur)  
	
		idx_list = np.concatenate(idx_list)
		self.y_list = np.concatenate(self.y_list)
		self.y0_index = idx_list[self.y_list == 0]
		self.y1_index = idx_list[self.y_list == 1]
		self.len = len(self.y1_index) * 2
		if self.type == "test":
			self.index_sampled = idx_list
			self.len = len(self.index_sampled)
		# self.sstride

	def _search_index(self, y, stride=200):
		# y : 1d array
		y_learn = y[np.arange(self.learning_win) + np.arange(0, len(y) - self.learning_win - self.pred_win - stride + 1, stride)[:, None]]  # n' x learning_win
		y_pred = y[np.arange(self.pred_win) + np.arange(self.learning_win, len(y) - self.pred_win - stride + 1, stride)[:, None]]  # n' x pred_win
		
		y_learn = np.any(y_learn == 1, axis=1)  # n'
		y_pred = np.any(y_pred == 1, axis=1)  # n'

		valid = np.logical_not(y_learn)  # n'
		idx_list = np.where(valid == True)[0] * stride  # Ensure it's a 1D array of indices
		y_pred_valid = y_pred[valid]
		return idx_list, y_pred_valid

	def __len__(self):
		return math.ceil(self.len / self.batch_size)

	def __getitem__(self, index):
		if index == 0 :
			self._reset()

		recs = []
		batch_idx = self.index_sampled[index * self.batch_size : min((index + 1) * self.batch_size, len(self.index_sampled))]
		if self.type == "train" : 
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
			recs.append(np.load(self.data_file_list[self.cur_page]))
			self.cur_page += 1

		recs = np.concatenate(recs, axis=0)  # Concatenate arrays
		recs = recs[np.arange(self.learning_win + self.pred_win) + batch_idx[:, None] - cur_page_idx] # batch_size x win_len x channel
		x = recs[:, :self.learning_win, :-2] # batch_size x learning_win_len x channel(waveforms)
		y_pred = np.any(recs[:, self.learning_win:, -1] == 1, axis=1).astype(np.float32) # batch_size
		y_all = np.any((recs[:, self.learning_win:, -2] < 95) == 1, axis=1)  # batch_size



		
		if self.type == "train" : 
			return x, y_pred
		else :
			return x, y_pred, y_all

	def _reset(self):
		self.cur_page = 0
		if not self.type == "test":
			self.index_sampled = np.sort(np.append(np.random.choice(self.y0_index, len(self.y1_index)), self.y1_index))



if __name__ =="__main__" :
	# Variables npy path list
	path_list = sorted(glob.glob(data_dir))
	train_list = sorted(path_list[:int(len(path_list) * 0.8)])
	validation_list = sorted(path_list[int(len(path_list) * 0.8):])



	WaveformsDataLoader("train", train_list, cfg)
	