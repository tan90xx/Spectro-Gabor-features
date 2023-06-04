import argparse
import config

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import h5py
import math
from librosa import fft_frequencies, frames_to_time, stft
from passivesonar import tpsw
from scipy.signal import spectrogram
import time
from scipy.signal import cheb2ord, cheby2, convolve, decimate, hilbert, lfilter, spectrogram

def pack_audio_files_to_hdf5(args):

    # Arguments & parameters
    h5_path = args.h5_path
    waveforms_hdf5_path = args.waveforms_hdf5_path

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    window_size = config.window_size
    hop_size = config.hop_size

    def load_h5(h5_path):
        # load training data
        with h5py.File(h5_path, 'r') as hf:
            print('List of arrays in input file:', list(hf.keys()))
            meta_dict = {
                'audio_name': hf['audio_name'][:], 
                'fold': hf['fold'][:], 
                'target': hf['target'][:], 
                'waveform': hf['waveform'][:].astype(np.float32)}
        return meta_dict
    
    meta_dict = load_h5(h5_path)
    audios_num=len(meta_dict['audio_name'])

    def cal_lofar(x, spectrum_bins_left=None):
        _, _, power = spectrogram(x,
                                window=('hann'),
                                nperseg=window_size,
                                noverlap=hop_size,
                                nfft=window_size,
                                fs=sample_rate,
                                detrend=False,
                                axis=0,
                                scaling='spectrum',
                                mode='magnitude')
        tpsw_signal = tpsw(power)
        # For stereo, without further changes, the genreated spectrogram has shape (freq, channel, time)
        if power.ndim == 3:  # temporary fix for stereo audio.
            power = power.mean(axis=1)
            power = power.squeeze()

        power = np.absolute(power)
        power = power / tpsw_signal  # , **tpsw_args)
        power = np.log10(power+1e-8)
        power[power < -0.2] = 0

        if spectrum_bins_left is None:
            spectrum_bins_left = int(power.shape[0]*0.8)
        power = power[:spectrum_bins_left, :]

        return power
    
    def cal_demon(x, max_freq=35, overlap_ratio=0.5):
        n_fft = config.window_size
        fs = config.sample_rate
        
        first_pass_sr = fs/25  # 31250/25

        q1 = round(fs/first_pass_sr)  # 25 for 31250 sample rate ; decimatio ratio for 1st pass
        q2 = round((fs/q1)/(2*max_freq))  # decimatio ratio for 2nd pass

        fft_over = math.floor(n_fft-2*max_freq*overlap_ratio)

        x = np.abs(x)  # demodulation

        x = decimate(x, q1, ftype='fir', zero_phase=False)
        x = decimate(x, q2, ftype='fir', zero_phase=False)
        final_fs = (fs//q1)//q2

        x /= x.max()
        x -= np.mean(x)
        sxx = stft(x,
                   window=('hann'),
                   win_length=n_fft,
                   hop_length=(n_fft - fft_over),
                   n_fft=n_fft)

        sxx = np.absolute(sxx)

        sxx = sxx / tpsw(sxx)

        return sxx[:410,:]
    
    def add_data_to_hdf5(hf, n):
        if n == 0:
            hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S80')
            hf.create_dataset('fold', shape=((audios_num,)), dtype=np.int32)
            hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.float32)
            hf.create_dataset('lofar', shape=((audios_num, 410, 61)), dtype=np.float32)#demon 61
        else:
            # Pack waveform & target of several audio clips to a single hdf5 file
            hf['audio_name'][n] = meta_dict['audio_name'][n]
            hf['fold'][n] = meta_dict['fold'][n]
            hf['target'][n] = meta_dict['target'][n]
            hf['lofar'][n] = cal_demon(meta_dict['waveform'][n])

    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        for n in range(audios_num):
            add_data_to_hdf5(hf, n)
            if (n % 20 == 0) or (n == 0):
                print(n)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--h5_path', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Directory of your workspace.')

    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')