from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import butter, cheby1, bessel, lfilter
from config import *
from sklearn import preprocessing
# ---------------------------------------处理1--------------------------------------------
def process1(y):
    #enhenc = Filter(spec_sub(y), FS / 8, FS, filter_type='Cheby1')
    enhenc = spec_sub(y)
    # 基本操作 减掉静音，补零
    wav = silence_filter(enhenc)
    #wav = mvn(wav)
    delta_sample = int(DT * FS)
    if wav.shape[0] < delta_sample:
        sample = np.zeros(shape=(delta_sample,), dtype=np.float32)
        index = int((delta_sample-wav.shape[0])//2)
        sample[index:index + wav.shape[0]] = wav
    return sample

# ---------------------------------------基本操作------------------------------------------
def not_silent(x, threshold=0.5):
    return np.sum(np.square(x)) > threshold


def mvn(data):
    data_zcore = data - np.mean(data)
    data_zcore = data_zcore / np.std(data)
    return data_zcore

# also MVN
def normalise(img):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,80))
    X_minmax = min_max_scaler.fit_transform(img)
    return X_minmax


def ola(X, seg_length):
    X = X.reshape((len(X) // seg_length, seg_length))
    length = int(X.shape[1] // 2)
    # left = np.concatenate([X[:-1,:length].flatten(),X[-1,:].flatten()],axis=0)
    right = np.concatenate([X[0, :].flatten(), X[1:, length:].flatten()], axis=0)
    return right

'''
def silence_filter(x, seg_length=1024, stride=512, silence_threshold=0.5):
    n = x.shape[0]
    x = mvn(x)
    x_segs = [x[i:i + seg_length] for i in range(0, n - seg_length, stride)]
    x_segs_filter = filter(lambda x: not_silent(x, silence_threshold), x_segs)
    x_rebuild = ola(np.array(list(x_segs_filter)).flatten(), seg_length)
    return x_rebuild
'''
# ttyadd: silence filter with energy threshold
def silence_filter(data,threshold=0.05):
    # discard the silence
    threshold_db =-10 * np.log10(threshold / 1.0)
    speak, _ = librosa.effects.trim(data, top_db=threshold_db, frame_length=2048, hop_length=1024)
    return speak
# ---------------------------------------谱减法------------------------------------------
def spec_sub(wav):
    # 计算 nosiy 带噪信号的频谱
    S_noisy = librosa.stft(wav, n_fft=256, hop_length=128, win_length=256)  # D x T
    #S_noisy = librosa.stft(wav, n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
    D, T = np.shape(S_noisy)
    Mag_noisy = np.abs(S_noisy)
    Phase_nosiy = np.angle(S_noisy)
    Power_nosiy = Mag_noisy ** 2
    # 估计噪声信号的能量
    # 由于噪声信号未知 这里假设 含噪（noisy）信号的前20帧为噪声
    Mag_nosie = np.mean(np.abs(S_noisy[:, :5]), axis=1, keepdims=True)
    Power_nosie = Mag_nosie ** 2
    Power_nosie = np.tile(Power_nosie, [1, T])
    ## 方法3 引入平滑
    Mag_noisy_new = np.copy(Mag_noisy)
    k = 1
    for t in range(k, T - k):
        Mag_noisy_new[:, t] = np.mean(Mag_noisy[:, t - k:t + k + 1], axis=1)
    Power_nosiy = Mag_noisy_new ** 2
    # 超减法去噪
    alpha = 100
    gamma = 1
    Power_enhenc = np.power(Power_nosiy, gamma) - alpha * np.power(Power_nosie, gamma)
    Power_enhenc = np.power(Power_enhenc, 1 / gamma)
    # 对于过小的值用 beta* Power_nosie 替代
    beta = 0.0001
    mask = (Power_enhenc >= beta * Power_nosie) - 0
    Power_enhenc = mask * Power_enhenc + beta * (1 - mask) * Power_nosie

    Mag_enhenc = np.sqrt(Power_enhenc)
    Mag_enhenc_new = np.copy(Mag_enhenc)
    # 计算最大噪声残差
    maxnr = np.max(np.abs(S_noisy[:, :5]) - Mag_nosie, axis=1)
    k = 1
    for t in range(k, T - k):
        index = np.where(Mag_enhenc[:, t] < maxnr)[0]
        temp = np.min(Mag_enhenc[:, t - k:t + k + 1], axis=1)
        Mag_enhenc_new[index, t] = temp[index]

    # 对信号进行恢复
    S_enhec = Mag_enhenc_new * np.exp(1j * Phase_nosiy)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)
    #sf.write(dst_path, enhenc, fs)
    return enhenc

# -----------------------------------------低通滤波-----------------------------------------------
def Filter(data, cutoff, fs=16000, order=8, filter_type='Cheby1'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if filter_type == 'Butter':
        b, a = butter(order, normal_cutoff)
    elif filter_type == 'Bessel':
        b, a = bessel(order, normal_cutoff)
    else:
        b, a = cheby1(order, 1, normal_cutoff)
    return lfilter(b, a, data)

# -------------------------------------------批处理----------------------------------------------
def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('$')[0], fn + '_{}.wav'.format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def split_wavs(src_root, dst_root, dt, rate,sub=False):
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir), ncols=50):
            src_fn = os.path.join(src_dir, fn)
            wav, rate = librosa.load(src_fn, rate)
            wav = librosa.effects.preemphasis(wav) #预加重
            if sub:
                wav = spec_sub(wav)
            wav = silence_filter(wav)
            wav = mvn(wav)
            delta_sample = int(dt * rate)
            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.float32)
                sample[:wav.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                continue

def read_dataset(inputfiles):
    file_list = []
    file_extensions = set(['.wav'])
    with open(inputfiles) as f:
        for line in f:
            filename = line.strip()
            ext = os.path.splitext(filename)[1]
            if ext in file_extensions:
                file_list.append(filename)
    return file_list

# -------------------------------------------画图------------------------------------------------
def draw_spec(*param, name=None, save=None, a=6, b=5, show=True, bar=False):
    # ------------------dft parameters---------------------
    FFT=16000
    HOP=160
    WIN=512
    FS=16000
    DT=2
    # ------------------ for color--------------------------
    COLOR="coolwarm"
    # ---------------- loop to draw ------------------------
    n = len(param)
    if n > 1:
        fig, axs = plt.subplots(1, n, sharey=True, figsize=(a, b))
        images = []
        for col, signal in enumerate(param):
            ax = axs[col]
            if np.ndim(param[0]) == 2:
                Mag_signal_db = signal
            else:
                S_signal = librosa.stft(signal, n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
                Mag_signal = np.abs(S_signal)
                Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

            Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
            X = np.arange(0,np.shape(Mag_signal_db)[1]/FS,1/FS)

            pcm = ax.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

            ax.set_xlim([0,np.shape(Mag_signal_db)[1]/FS])
            images.append(pcm)
            ax.label_outer()
            ax.set_ylim([1, int(FFT / 2 + 1)])
            ax.set_xlabel('Time (s)')
            ax.grid(False)
            if name:
                ax.set_title("{}".format(name[col]))
            if col == 0:
                ax.set_ylabel('Frequency (Hz)')

        # Find the min and max of all colors for use in setting the color scale.
        if bar:
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs, orientation='horizental', fraction=.05, label='Magnitude(dB)')

    else:
        plt.figure(figsize=(a, b))

        if np.ndim(param[0]) == 2:
            Mag_signal_db = param[0]
        else:
            S_signal = librosa.stft(param[0], n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
            Mag_signal = np.abs(S_signal)
            Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

        Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
        X = np.arange(0,np.shape(Mag_signal_db)[1]/FS,1/FS)

        pcm = plt.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

        plt.xlim([0, np.shape(Mag_signal_db)[1] / FS])

        if name:
            plt.title("{}".format(name[0]))
        plt.ylim([0, int(FFT / 2 + 1)])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(False)
        plt.colorbar(pcm, label='Magnitude(dB)')
    if save:
        plt.savefig('{}.png'.format(save), dpi=600, bbox_inches="tight")
    if show:
        plt.show()

def plot_signals_time(titles, signals, nrows=3, ncols=5):
    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 6))

    z = 0
    for i in range(nrows):
        for y in range(ncols):
            ax[i, y].set_title(titles[z])
            ax[i, y].plot(signals[z])
            ax[i, y].set_xticks([])
            ax[i, y].set_yticks([])
            ax[i, y].grid(False)
            z += 1

    plt.show()


def plot_spectrogram(titles, signals, title, shape=(16, 8), nrows=3, ncols=5):
    fig, ax = plt.subplots(nrows, ncols, figsize=shape)
    fig.suptitle(title, size=20)
    plt.set_cmap('viridis')

    z = 0
    for i in range(nrows):
        for y in range(ncols):
            ax[i, y].set_title(titles[z])
            ax[i, y].imshow(signals[z].squeeze(),cmap=COLOR)
            ax[i, y].set_ylim([0, int(np.shape(signals[z].squeeze())[0])])
            ax[i, y].set_xticks([])
            ax[i, y].set_yticks([])
            ax[i, y].grid(False)
            z += 1

    plt.show()
