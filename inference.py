import torch
import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

from models.model import Model
from main import Trainer, training_params


def load_model(state_dict):
    training_params['test'] = True
    training_params['load_pretrained'] = False
    training_params['device'] = device
    trainer = Trainer(**training_params)
    model = trainer.model
    model.load_state_dict(torch.load(state_dict))
    return model


def load_audio(file_path, sample_rate, num_time_samples):
    signal, _ = librosa.load(file_path, sr=sample_rate)
    signal = librosa.util.normalize(signal)
    signal = signal[:num_time_samples]
    return signal


def generate_audio(signal, model, device):
    x = torch.FloatTensor(signal)
    x.to(device)
    x_hat, _ = model(x.view(1, 1, -1))
    x_hat = x_hat.view(-1)
    return x_hat.detach().cpu().numpy()


def write_audio_to_file(file_name, audio):
    write('test.wav', sample_rate, audio)


def display_spectrogram(signal, ax, title):
    stft = librosa.core.stft(signal, n_fft=2048, hop_length=512)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    librosa.display.specshow(log_spectrogram, ax=ax)
    ax.set_title(title)


def display_raw_audio(signal, ax, title):
    ax.plot(signal)
    ax.set_title(title)


def visualize_audio(x, x_hat, epoch):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    display_spectrogram(x, axes[0, 0], 'spectrogram of original audio')
    display_spectrogram(x_hat, axes[0, 1], 'spectrogram of generated audio')
    display_raw_audio(x, axes[1, 0], 'original audio')
    display_raw_audio(x_hat, axes[1, 1], 'generated audio')
    fig.suptitle(f'Epoch {epoch}')
    plt.show()
    

if __name__ == '__main__':
    sample_rate = 16384
    num_time_samples = sample_rate * 2
    device = torch.device('cuda')
    epoch = 20

    model = load_model(f'trained_models\\saved_model_epoch_{epoch}')
    x = load_audio('afrikaans1.wav', sample_rate, num_time_samples)
    x_hat = generate_audio(x, model, device)
    print(x_hat)
    write_audio_to_file('test.wav', x_hat)
    visualize_audio(x, x_hat, epoch)
