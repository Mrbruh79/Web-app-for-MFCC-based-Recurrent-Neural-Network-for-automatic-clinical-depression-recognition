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

#Creating datasets to store processed data
train_data = pd.DataFrame({"Audio":[],"Binary":[] ,"Score":[]})
validation_data = pd.DataFrame({"Audio":[],"Binary":[] ,"Score":[]})
test_data = pd.DataFrame({"Audio":[],"Binary":[] ,"Score":[]})

def MFCC_Preprocess(audio):
    normalized_audio = effects.normalize(audio)
    fs = audio.frame_rate
    data = audio.get_array_of_samples()
    window_length = 2.5  # seconds
    step_size = 0.5  # seconds
    window_samples = int(window_length * fs)
    step_samples = int(step_size * fs)
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
        
    mel_filter = cp.asarray(librosa.filters.mel(fs, window_samples, n_mels=n_mels))
        
    mel_spectrum = cp.dot(mel_filter, fft_magnitude)
    smoothed_spectrum = cp.log(mel_spectrum)

    # Perform Karhunen-Loeve (KL) transform (approximated by DCT)
    kl_transform = cp.asarray(scipy.fftpack.dct(cp.asnumpy(mel_spectrum), axis=0, norm='ortho'))

    # Obtain cepstral features
    cepstral_coefficients = cp.dot( cp.cos(cp.pi * cp.arange(n_mels) * cp.arange(n_ceps).reshape(-1, 1) / n_mels),kl_transform)
    return cepstral_coefficients
  
  def augumentation(audio):
    sr = audio.frame_rate
    y = np.array(audio.get_array_of_samples())
    
    #Noise Injection
    y1 = y + np.random.normal(0, 0.05, len(y))
    #Pitch augmenter
    y2 = librosa.effects.pitch_shift(y.astype(np.float64), sr, n_steps=1.5)
    #shift augmenter 
    shift = np.random.uniform(-0.2, 0.2)
    y3 = librosa.effects.time_stretch(y.astype(np.float64), 1+shift)
    if shift < 0:
        y3[:int(-shift*sr)] = 0
    else:
        y3[-int(shift*sr):] = 0
    #speed augmenter
    y4 = librosa.effects.time_stretch(y.astype(np.float64), 1.5)
    
    #Converting back to audio segment
    y1[np.isnan(y1)] = 0
    y2[np.isnan(y2)] = 0
    y3[np.isnan(y3)] = 0
    y4[np.isnan(y4)] = 0
    y1[np.isinf(y1)] = 0
    y2[np.isinf(y2)] = 0
    y3[np.isinf(y3)] = 0
    y4[np.isinf(y4)] = 0
    y1 = audio._spawn(np.clip(y1.astype(int), -32768, 32767))
    y2 = audio._spawn(np.clip(y2.astype(int), -32768, 32767))
    y3 = audio._spawn(np.clip(y3.astype(int), -32768, 32767))
    y4 = audio._spawn(np.clip(y4.astype(int), -32768, 32767))
    return [audio,y1,y2,y3,y4]
  
# Making training dataset
dir = r"diac-dataset/{num}_P"
out_dir = r"/{num}_P"
k = 0
for index,row in train_voice.iterrows(): 
    curdir = dir.format(num = str(int(row["Participant_ID"])))
    if os.path.exists(curdir):
        #Importing Audio file
        aud_dir = curdir +"/" + str(int(row["Participant_ID"])) + "_AUDIO.wav"
        audio = AudioSegment.from_file(aud_dir)
        #Importing transcription data for the audio file 
        data_dir = curdir +"/" + str(int(row["Participant_ID"])) + "_TRANSCRIPT.csv"
        df = pd.read_csv(data_dir ,delimiter = "\t")
        #creating final audio files
        desired_section = AudioSegment.empty()

        for index, r in df.iterrows():
            #Extracting only participant audio from the file 
            if(r["speaker"] == "Participant"):
                desired_section = desired_section + audio[r["start_time"] * 1000 :r["stop_time"] * 1000]

        #Applying Augumentation 
        audiofiles = augumentation(desired_section)
        for aud in audiofiles:
            metrics = MFCC_Preprocess(aud)
            train_data = train_data.append(pd.DataFrame({"Audio":[metrics],"Binary":[row["PHQ8_Binary"]] ,"Score":[row["PHQ8_Score"]]}), ignore_index=True)
train_data.to_csv(r'/kaggle/working/train.csv')

# Making validation dataset
dir = r"diac-dataset/{num}_P"
out_dir = r"/{num}_P"
for index,row in dev_voice.iterrows(): 
    curdir = dir.format(num = row["Participant_ID"])
    if os.path.exists(curdir):
        #Importing Audio file
        aud_dir = curdir +"/" + str(row["Participant_ID"]) + "_AUDIO.wav"
        audio = AudioSegment.from_file(aud_dir)
        #Importing transcription data for the audio file 
        data_dir = curdir +"/" + str(row["Participant_ID"]) + "_TRANSCRIPT.csv"
        df = pd.read_csv(data_dir ,delimiter = "\t")
        #creating final audio files
        desired_section = AudioSegment.empty()

        for index, r in df.iterrows():
            #Extracting only participant audio from the file 
            if(r["speaker"] == "Participant"):
                desired_section = desired_section + audio[r["start_time"] * 1000 :r["stop_time"] * 1000]

        #Applying Augumentation 
        audiofiles = augumentation(desired_section)
        for aud in audiofiles:
            metrics = MFCC_Preprocess(aud)
            validation_data = validation_data.append(pd.DataFrame({"Audio":[metrics],"Binary":[row["PHQ8_Binary"]] ,"Score":[row["PHQ8_Score"]]}))
validation_data.to_csv(r'/kaggle/working/validation.csv')  

#Making test dataset
dir = r"/kaggle/input/diac-dataset/{num}_P"
out_dir = r"/kaggle/working/{num}_P"
for index,row in test_voice.iterrows():
    curdir = dir.format(num = row["participant_ID"])
    if os.path.exists(curdir):
        #Importing Audio file
        aud_dir = curdir +"/" + str(row["participant_ID"]) + "_AUDIO.wav"
        audio = AudioSegment.from_file(aud_dir)
        #Importing transcription data for the audio file 
        data_dir = curdir +"/" + str(row["participant_ID"]) + "_TRANSCRIPT.csv"
        df = pd.read_csv(data_dir ,delimiter = "\t")
        #creating final audio files
        desired_section = AudioSegment.empty()

        for index, r in df.iterrows():
            #Extracting only participant audio from the file 
            if(r["speaker"] == "Participant"):
                desired_section = desired_section + audio[r["start_time"] * 1000 :r["stop_time"] * 1000]

        #Applying Augumentation 
        audiofiles = augumentation(desired_section)
        for aud in audiofiles:
            metrics = MFCC_Preprocess(aud)
           #Making test dataset
dir = r"/kaggle/input/diac-dataset/{num}_P"
out_dir = r"/kaggle/working/{num}_P"
for index,row in test_voice.iterrows():
    curdir = dir.format(num = row["participant_ID"])
    if os.path.exists(curdir):
        #Importing Audio file
        aud_dir = curdir +"/" + str(row["participant_ID"]) + "_AUDIO.wav"
        audio = AudioSegment.from_file(aud_dir)
        #Importing transcription data for the audio file 
        data_dir = curdir +"/" + str(row["participant_ID"]) + "_TRANSCRIPT.csv"
        df = pd.read_csv(data_dir ,delimiter = "\t")
        #creating final audio files
        desired_section = AudioSegment.empty()

        for index, r in df.iterrows():
            #Extracting only participant audio from the file 
            if(r["speaker"] == "Participant"):
                desired_section = desired_section + audio[r["start_time"] * 1000 :r["stop_time"] * 1000]

        #Applying Augumentation 
        audiofiles = augumentation(desired_section)
        for aud in audiofiles:
            metrics = MFCC_Preprocess(aud)
            test_data = test_data.append(pd.DataFrame({"Audio":[metrics],"Binary":[full_split.loc[full_split['Participant_ID'] == row["participant_ID"]]["PHQ_Binary"]] ,"Score":[full_split.loc[full_split['Participant_ID'] == row["participant_ID"]]["PHQ_Score"]]}))
test_data.to_csv(r'/kaggle/working/test.csv')

  
