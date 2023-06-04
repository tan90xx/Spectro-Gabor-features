import os
import numpy as np
import argparse
import h5py
import librosa
import time

import config
from utilities import create_folder, traverse_folder, wash_folder, float32_to_int16
from sklearn import preprocessing

def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target

'''
def pad_spilt_sequence(x, seg_len):
    audio_len = len(x)
    pad_len = seg_len - audio_len%seg_len
    if pad_len < seg_len:
        audio_padded = np.concatenate((x, np.zeros(pad_len)), axis=0)
    else:
        audio_padded = x
    audio_segs = audio_padded.reshape((-1, seg_len))
    return audio_segs
'''
def pad_spilt_sequence(x, d, s):
    patches = list()
    if len(x) < d:
        pad_len = d - len(x)
        patches.append(np.concatenate((x, np.zeros(pad_len)), axis=0))
    else:
        max_i = len(x) - d + 1
        for i in range(0, max_i, s):
            patch = x[i : i+d]
            patches.append(patch)
    scaler = preprocessing.StandardScaler()
    patches_scaler = scaler.fit_transform(patches)
    return np.array(patches_scaler)


def pack_audio_files_to_hdf5(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    stride_samples = config.stride_samples
    clip_samples = config.clip_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    kfold = config.kfold

    # Paths
    audios_dir = os.path.join(dataset_dir)

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'data', 'minidata_waveform_10.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'data', 'waveform.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    # for DeepShip
    (audio_names, audio_paths) = traverse_folder(audios_dir)

    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[audio_path.split(os.path.sep)[-3]] for audio_path in audio_paths]), 
        'fold': np.arange(len(audio_names)) % kfold + 1}
    '''
    # for ShipsEar
    (audio_names, audio_paths) = wash_folder(audios_dir)
    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[config.ShipsEar.get(audio_name)] for audio_name in audio_names]), 
        'fold': np.arange(len(audio_names)) % kfold + 1}
    '''
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    def add_data_to_hdf5(hf, audio, n):
        audio_name = meta_dict['audio_name'][n]
        target = meta_dict['target'][n]
        fold = meta_dict['fold'][n]

        audio_name = np.array(audio_name.encode())
        target = to_one_hot(meta_dict['target'][n], classes_num)
        fold = np.array(fold)

        m = len(audio)
        audio_name, target, fold = audio_name.repeat(m), target.repeat(m), fold.repeat(m)
        target = target.reshape(classes_num,-1).transpose(1,0)

        if n == 0:
            # create the hdf5 file
            hf.create_dataset(name='audio_name', 
                                shape=np.array(audio_name).shape, 
                                maxshape=(None,),
                                dtype='S80')

            hf.create_dataset(name='waveform', 
                                shape=np.array(audio).shape, 
                                maxshape=(None, clip_samples),
                                dtype=np.int16)

            hf.create_dataset(name='target', 
                                shape=np.array(target).shape, 
                                maxshape=(None, classes_num),
                                dtype=np.float32)

            hf.create_dataset(name='fold', 
                                shape=np.array(fold).shape, 
                                maxshape=(None,),
                                dtype=np.int32)

            hf['audio_name'][...] = audio_name
            hf['waveform'][...] = float32_to_int16(audio)
            hf['target'][...] = target
            hf['fold'][...] = fold     
        else:
            # print("patches num:", m)
            hf['audio_name'].resize([hf['audio_name'].shape[0]+m,])
            hf['waveform'].resize([hf['waveform'].shape[0]+m, clip_samples])
            hf['target'].resize([hf['target'].shape[0]+m, classes_num])
            hf['fold'].resize([hf['fold'].shape[0]+m,])

            hf['audio_name'][-m:] = audio_name
            hf['waveform'][-m:] = float32_to_int16(audio)
            hf['target'][-m:] = target
            hf['fold'][-m:] = fold

    audios_num = len(meta_dict['audio_name'])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        for n in range(audios_num):
            print(n)
            audio_path = meta_dict['audio_path'][n]
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            audio_clips = pad_spilt_sequence(audio, clip_samples, stride_samples)
            add_data_to_hdf5(hf, audio_clips, n)

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')