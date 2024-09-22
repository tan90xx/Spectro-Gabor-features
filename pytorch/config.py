import numpy as np

sample_rate = 16000
clip_samples = sample_rate * 5
kfold = 4

labels = ['Cargo', 'Passengership', 'Tanker', 'Tug']
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)

#indexes = np.load(file='/root/autodl-fs/可视化/indexes.npy')
#------------------------------------------------------
mel_bins = 64
fmin = 0
fmax = None
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None