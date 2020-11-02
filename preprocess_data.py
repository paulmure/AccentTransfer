from pydub import AudioSegment
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import random
import librosa
import pickle
import re
import os
import torch

# we expect these folders to exist before running this file
mp3_dir = os.path.join('data', 'recordings_mp3')
wav_dir = os.path.join('data', 'recordings_wav')

# audio processing parameters
sample_rate = 16384
sample_window = sample_rate * 2  # we want 2 second clips


def extract_label_from_filename(filename):
    return re.split(r'[0-9]+', filename)[0]


def convert_mp3_to_wav(filename):
    sound = AudioSegment.from_mp3(os.path.join(mp3_dir, filename))
    sound.export(os.path.join(wav_dir, f'{filename[:-4]}.wav'), format='wav')


def slice_data(signal, label):
    num_windows = signal.shape[0] // sample_window

    res = [(label, signal[i * sample_window : (i + 1) * sample_window]) \
                for i in range(num_windows)]

    # pad data left over at the end with 0s
    if signal.shape[0] > num_windows * sample_window:
        pad_length = (num_windows + 1) * sample_window - signal.shape[0]
        padded_clip = np.pad(signal[num_windows * sample_window:], (0, pad_length), 'constant', constant_values=(0, 0))
        res.append((label, padded_clip))
    
    return res


def convert_wav_to_wave_data(filename):
    label = extract_label_from_filename(filename)
    signal, _ = librosa.load(os.path.join(wav_dir, filename), sr=sample_rate)
    return slice_data(signal, label)


def convert_directory(func, directory):
    filenames = os.listdir(directory)
    random.shuffle(filenames)
<<<<<<< HEAD
    filenames = filenames[:50]
=======
    filenames = filenames[:1700]
>>>>>>> 7ae88f449f05e96a568209afb01f3e909bfac985
    with Pool() as pool:
        return list(tqdm(pool.imap(func, filenames)))


def format_data(data):
    flattened_data = [item for sublist in data for item in sublist]

    labels = list(set(label for label, _ in flattened_data))
    labels_to_idx = {labels[i]: i for i in range(len(labels))}
    idx_to_labels = {v: k for k, v in labels_to_idx.items()}

    final_data = [(torch.tensor(audio, dtype=torch.float32), labels_to_idx[label]) for label, audio in flattened_data]
    return final_data, idx_to_labels


if __name__ == '__main__':
    # convert_directory(convert_mp3_to_wav, mp3_dir)
    print("Reading audio data from wav files...")
    raw_data = convert_directory(convert_wav_to_wave_data, wav_dir)
    print("Creating torch tensors...")
    final_data, idx_to_labels = format_data(raw_data)

    print("Dumping data to disk...")
    with open('audio_data', 'wb') as f:
        pickle.dump(final_data, f)
    with open('class_label_names', 'wb') as f:
        pickle.dump(idx_to_labels, f)

    print("All done!")
