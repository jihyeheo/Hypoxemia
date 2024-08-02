import numpy as np

path = "./processed/P2_200619_103025.vital.npy"
data = np.load(path, allow_pickle=True)
print(data.shape)