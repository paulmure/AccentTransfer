from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile
from multiprocessing import Pool
import pickle
import os

mp3_dir = os.path.join('data', 'recordings_mp3')
wav_dir = os.path.join('data', 'recordings_wav')


def convert_mp3_to_wav(filename):
    sound = AudioSegment.from_mp3(os.path.join(mp3_dir, filename))
    sound.export(os.path.join(wav_dir, f'{filename[:-4]}.wav'), format='wav')


def convert_wav_to_spectrogram(filename):
    sample_rate, samples = wavfile.read(os.path.join(wav_dir, filename))
    _, _, spectrogram = signal.spectrogram(samples, sample_rate)
    return filename, spectrogram


def convert_directory(func, directory):
    filenames = os.listdir(directory)
    with Pool() as pool:
        return pool.map(func, filenames)


if __name__ == "__main__":
    convert_directory(convert_mp3_to_wav, mp3_dir)
    spectrograms = convert_directory(convert_wav_to_spectrogram, wav_dir)
    with open('spectrogram_data', 'wb') as f:
        pickle.dump(spectrograms, f)
