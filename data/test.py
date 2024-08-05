import numpy as np
import matplotlib.pyplot as plt

path = "./processed/SNUH/train/P1_190524_094718.vital.npy"
data = np.load(path, allow_pickle=True)
print(data.shape)



for ii in range(12) :
    plt.subplot(12, 1, ii+1)
    plt.plot(data[:, ii].reshape(-1,))
plt.savefig("data_check.png")
