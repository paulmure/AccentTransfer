from multiprocessing import Pool
import librosa
import pickle
import os
from torch import tensor

DATA_DIR = os.path.join('data', 'raw')
OUT_PUT_DIR = os.path.join('data', 'final')

# audio processing parameters
sample_rate = 16384
sample_window = sample_rate * 2  # we want 2 second clips


def list_data():
    """
    List all the data in a list of tuples (label_idx, file_path)
    
    Also create a idx_to_labels file
    """
    top_level_data_dir = os.listdir(DATA_DIR)
    num_classes = len(top_level_data_dir)
    idx_to_labels = {i: top_level_data_dir[i] for i in range(num_classes)}
    with open(os.path.join(OUT_PUT_DIR, 'idx_to_labels'), 'wb') as f:
        pickle.dump(idx_to_labels, f)
    
    data_list = []
    for i in range(len(top_level_data_dir)):
        speakers_dir = os.path.join(DATA_DIR, top_level_data_dir[i])
        speakers = os.listdir(speakers_dir)
        for speaker in speakers:
            files_dir = os.path.join(speakers_dir, speaker)
            file_names = os.listdir(files_dir)
            for file_name in file_names:
                data_list.append((i, os.path.join(files_dir, file_name)))
    return idx_to_labels, data_list


def slice_data(signal):
    if signal.shape[0] < sample_window: return None

    num_windows = signal.shape[0] // sample_window

    res = [signal[i * sample_window : (i + 1) * sample_window] \
            for i in range(num_windows)]

    # append left over data at the end
    if signal.shape[0] > num_windows * sample_window:
        res.append(signal[-sample_window:])
    
    return res


def get_audio_slices(file_path):
    signal, _ = librosa.load(file_path, sr=sample_rate)
    return slice_data(signal)


def convert(label, file_path, output_idx):
    audio_clips = get_audio_slices(file_path)
    if audio_clips is None: return

    for audio_clip in audio_clips:
        mfcc = librosa.feature.mfcc(audio_clip, n_fft=2048, hop_length=512, n_mfcc=30)
        audio_tensor = tensor(audio_clip).float()
        mfcc_tensor = tensor(mfcc).float()
        final_data = (audio_tensor, mfcc_tensor, label)
        with open(os.path.join(OUT_PUT_DIR, 'data', str(output_idx)), 'wb') as f:
            pickle.dump(final_data, f)


def convert_data(data):
    data = [(data[i][0], data[i][1], i) for i in range(len(data))]
    with Pool() as pool:
        pool.starmap(convert, data)


if __name__ == '__main__':
    print("Reading list of raw data files...")
    idx_to_labels, data = list_data()

    print("Converting data to tensors...")
    convert_data(data)

    print("All done!")
