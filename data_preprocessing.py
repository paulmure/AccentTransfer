from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from multiprocessing import Pool
import os

mp3_dir = os.path.join('data', 'recordings_mp3')
wav_dir = os.path.join('data', 'recordings_wav')


def convert_mp3_to_wav(filename):
    sound = AudioSegment.from_mp3(os.path.join(mp3_dir, filename))
    sound.export(os.path.join(wav_dir, f'{filename[:-4]}.wav'), format='wav')


def convert_data_to_wav():
    filenames = os.listdir(mp3_dir)
    with Pool() as pool:
        pool.map(convert_mp3_to_wav, filenames)


def test():
    dim_set = set()

    for filename in os.listdir(wav_dir):
        sample_rate, samples = wavfile.read(os.path.join(wav_dir, filename))
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        dim_set.add(spectrogram.shape)

    print(dim_set)


if __name__ == "__main__":
    # convert_data_to_wav()
    test()
