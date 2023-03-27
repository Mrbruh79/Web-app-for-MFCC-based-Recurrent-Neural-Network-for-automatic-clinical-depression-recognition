
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

    while end_idx <= len(data):
        # Extract segment of audio data
        segment = data[start_idx:end_idx]
        # Apply windowing function
        window = np.hanning(window_samples)
        windowed_segment = segment * window

        # Compute DFT of windowed segment
        dft = np.fft.rfft(windowed_segment)

        # Retain only the logarithmic of magnitude of the DFT
        dft_magnitude = np.log(np.abs(dft))
        dft_magnitude[np.isnan(dft_magnitude)] = 0
        dft_magnitude[np.isinf(dft_magnitude)] = 0
        # Smooth the spectrum to emphasize perceptually meaningful frequencies
        fft_magnitude = librosa.effects.harmonic(dft_magnitude)
        
        mel_filter = librosa.filters.mel(fs, window_samples, n_mels=n_mels)
        
        mel_spectrum = np.dot(mel_filter, fft_magnitude)
        smoothed_spectrum = np.log(mel_spectrum)

        # Perform Karhunen-Loeve (KL) transform (approximated by DCT)
        kl_transform = scipy.fftpack.dct(mel_spectrum, axis=0, norm='ortho')

        # Obtain cepstral features
        cepstral_coefficients = np.dot( np.cos(np.pi * np.arange(n_mels) * np.arange(n_ceps).reshape(-1, 1) / n_mels),kl_transform)
        
        feature_vectors.append(cepstral_coefficients)

        # Update start and end indices for next window
        start_idx += step_samples
        end_idx += step_samples
    
    # stack all the feature vectors
    feature_vectors = np.vstack(feature_vectors)
    # calculate the mean and standard deviation across the coefficients
    mean = np.mean(feature_vectors, axis=0)
    std = np.std(feature_vectors, axis=0)

    # standardize the feature vectors by subtracting the mean and dividing by the standard deviation
    feature_vectors = (feature_vectors - mean) / std
    return feature_vectors
