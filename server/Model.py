import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import ast
from pydub import effects
import scipy
from scipy.io import wavfile
from scipy.fftpack import fft
from pydub import AudioSegment
from IPython.display import Audio
import librosa
import shutil
from keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense
from keras.regularizers import l1
import keras.optimizers
from keras.models import Model
from numba import cuda
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
import tensorflow.keras.optimizers
import tensorflow as tf
np.set_printoptions(threshold=sys.maxsize) 
threshold = 0.5
from tensorflow.keras.models import model_from_json

args = sys.argv


def MFCC_Preprocess(audio):
    normalized_audio = effects.normalize(audio)
    fs = audio.frame_rate
    data = audio.get_array_of_samples()
    window_length = 2.5  # seconds
    step_size = 0.5  # seconds
    window_samples = int(window_length * fs)
    step_samples = int(step_size * fs)
    # print(len(data) , " ",len(data)/step_samples)
    # Define number of Mel-frequency bins
    n_mels = 24

    # Define number of cepstral features
    n_ceps = 60

    # Initialize start and end indices for windowing
    start_idx = 0
    end_idx = window_samples

    # Initialize list to store feature vectors
    feature_vectors = []

    with ThreadPoolExecutor() as executor:
        futures = []
        while end_idx <= len(data):
            # Extract segment of audio data
            segment = data[start_idx:end_idx]
            segment = cp.array(segment)
            futures.append(executor.submit(feature_extraction_kernel,segment,window_samples,fs,n_mels,n_ceps))
            start_idx += step_samples
            end_idx += step_samples
        for future in futures:
            feature_vectors.append(future.result())
    feature_vectors = np.vstack(feature_vectors)
    # calculate the mean and standard deviation across the coefficients
    mean = cp.mean(feature_vectors, axis=0)
    std = cp.std(feature_vectors, axis=0)
    feature_vectors = (feature_vectors - mean) / std
    return cp.asnumpy(feature_vectors)

def feature_extraction_kernel(segment,window_samples,fs,n_mels,n_ceps):
    # Apply windowing function
    window = cp.hanning(window_samples)
    windowed_segment = segment * window

    # Compute DFT of windowed segment
    dft = cp.fft.rfft(windowed_segment)

    # Retain only the logarithmic of magnitude of the DFT
    dft_magnitude = cp.log(cp.abs(dft))
    dft_magnitude[cp.isnan(dft_magnitude)] = 0
    dft_magnitude[cp.isinf(dft_magnitude)] = 0
    # Smooth the spectrum to emphasize perceptually meaningful frequencies
    fft_magnitude = cp.asarray(librosa.effects.harmonic(cp.asnumpy(dft_magnitude)))
        
    mel_filter = cp.asarray(librosa.filters.mel(sr=fs, n_fft=window_samples, n_mels=n_mels))
        
    mel_spectrum = cp.dot(mel_filter, fft_magnitude)
    smoothed_spectrum = cp.log(mel_spectrum)

    # Perform Karhunen-Loeve (KL) transform (approximated by DCT)
    kl_transform = cp.asarray(scipy.fftpack.dct(cp.asnumpy(mel_spectrum), axis=0, norm='ortho'))

    # Obtain cepstral features
    cepstral_coefficients = cp.dot( cp.cos(cp.pi * cp.arange(n_mels) * cp.arange(n_ceps).reshape(-1, 1) / n_mels),kl_transform)
    
    return cepstral_coefficients


sys.path.append(r"C:\Users\Restandsleep\Downloads\Web-app-for-MFCC-based-Recurrent-Neural-Network-for-automatic-clinical-depression-recognition-main\server")
audio = AudioSegment.from_file(r"uploads/300.wav")
proc = MFCC_Preprocess(audio)
proc = tf.keras.preprocessing.sequence.pad_sequences(np.array([proc]), padding='pre')


json_file = open(r'./model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"./mod.h5")
optimizer = tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.0015, decay=1e-6)
loaded_model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
              metrics=['accuracy'])
# loaded_model.summary()
score = loaded_model.predict(proc, verbose=0)
depression = False
level = 0
if(score[0][0]>threshold):
    depression = 1
else:
    depression = 0

level = score[1][0].argmax()

print("{" + f'"depressed":{depression},"level":{level}' + "}")
