# ------------------ prepdata --------------------------
sample_rate = 16000
stride_samples = sample_rate * 15
clip_samples = sample_rate * 30
kfold = 4

labels = ['Cargo', 'Passengership', 'Tanker', 'Tug']
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)
# ------------------dft parameters---------------------
FFT = 16000
HOP = 160
WIN = 512
FS = 16000
DT = 2
# ------------------ for color--------------------------
COLOR="coolwarm"
