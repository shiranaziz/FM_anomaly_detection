import scipy.io
import numpy as np
import pandas as pd

data = scipy.io.loadmat("data/arrhythmia.mat")
samples = data['X']  # 452
labels = ((data['y']).astype(np.int32)).reshape(-1)

norm_samples = samples[labels == 0]  # 386 norm
anom_samples = samples[labels == 1]  # 66 anom

n_train = len(norm_samples) // 2
x_train = norm_samples[:n_train]  # 193 train

val_real = norm_samples[n_train:]
val_fake = anom_samples
