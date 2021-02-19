import librosa as lb
from librosa.display import specshow
from librosa import griffinlim
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def load_test_song():
    y, sr = lb.load(r'../data/songs/dbstf_1.wav')
    print("sr:", sr)
    y = y[-500000:]
    return y, sr


def spectrogram(y, sr, to_db=False):
    print("melspectrogram...")
    M = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    print('m:', M.shape)
    if to_db:
        M = lb.power_to_db(M, ref=np.max)
    return M


def mfcc(M):
    return lb.feature.mfcc(S=M)


def chroma():
    chroma = lb.feature.chroma_stft()
    return chroma


def show(X, title=''):
    fig, ax = plt.subplots()
    img = specshow(X, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=title)
    plt.show()


def mel_to_audio(M, sr):
    print("mel_to_stft...")
    stft = lb.feature.inverse.mel_to_stft(M, sr=sr)
    print("griffinlim...")
    y_inv = griffinlim(stft, hop_length=512)
    return y_inv


def write_audio(y, sr):
    print('writing audio...')
    sf.write(r'../data/songs/test_inv.wav', y, sr)


if __name__ == '__main__':
    y, sr = load_test_song()
    print('done')
