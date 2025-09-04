import librosa
import soundfile as sf
from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt

def equalizer(y,sr, low_gain=0.0, mid_gain=0.0, high_gain=0.0):
    # y =  NumPy array containing the audio time series (the amplitude values)
    # sr = sample rate
    #modify the amplitude of different frequency bands
        #low_gain: bass
        # mid_gain: mids
        # high_gain: boost treble
    
    #stft = short-time fourier transform
    #You know the frequency content of the signal changes over time
    #Find which frequencies are present at each moment in time
    #You get a complex-valued 2D matrix D:
        # Rows = frequency bins, Columns = time frames
    y = np.asarray(y, dtype=np.float32)
    D = librosa.stft(y)
    
    #extract amplitude and phase of the STFT original signal (from each index)
    S, phase = np.abs(D), np.angle(D)
        
    #tells you frequency (for each rows). map each row index (frequency bins) to real frequency values,
    freqs = librosa.fft_frequencies(sr=sr)
    
    #Filter out frequencies using Boolean mask
    #select rows that belong to the approprate frequency bin
    low_band = (freqs < 400)# bass and sub-bas, kick drums, bass guitar: below 400 Hz
    mid_band = (freqs >= 400) & (freqs < 3000)#  vocal clarity, snare drums, guitar, piano, and other intruments: 400 Hz – 3 kHz
    high_band = (freqs >= 3000)# treble:above 3 kHz
    
    # Convert dB gains to linear scale
    low_gain_lin = 10**(low_gain / 20)
    mid_gain_lin = 10**(mid_gain / 20)
    high_gain_lin = 10**(high_gain / 20)
    
    # Apply amlitude gains to each band of original signal
    S[low_band, :] *= low_gain_lin
    S[mid_band, :] *= mid_gain_lin
    S[high_band, :] *= high_gain_lin
    
    # Reconstruct STFT with modified magnitude and original phase
    D_eq = S * np.exp(1j * phase)

    # Inverse STFT to time domain
    y_eq = librosa.istft(D_eq)
    
    # #Plot BEFORE filtering
    # plt.figure()
    # S_db = librosa.amplitude_to_db(abs(D), ref=1.0)  # Use ref=1.0 to get absolute dB scale
    # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', vmin=-30, vmax=30)
    # cbar = plt.colorbar()
    # cbar.set_label("Amplitude (dB)")
    # plt.title("Spectrogram Before Filtering")
    # plt.show()

    #Plot AFTER filtering
    # plt.figure()
    # D_after = librosa.stft(y_eq)
    # S_af = np.abs(D_after)
    # S_db_af = librosa.amplitude_to_db(S_af, ref=1.0)
    # librosa.display.specshow(S_db_af, sr=sr, x_axis='time', y_axis='hz', vmin=-30, vmax=30)
    # cbar = plt.colorbar()
    # cbar.set_label("Amplitude (dB)")
    # plt.title("Spectrogram After Filtering")
    # plt.show()
    
    return y_eq, sr
    