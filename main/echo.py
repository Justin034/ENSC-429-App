import librosa
import soundfile as sf
from IPython.display import Audio
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

def echo(y, sr):

    #Defining multiple echoes
    delay_seconds = [0.5, 0.6, 0.7] #in seconds
    amplitudes = [0.5, 0.3, 0.1] #corresponding amplitudes of the echoes

    delay_samples = [int(sr * d) for d in delay_seconds] #how many samples to delay by

    #Create impulse response
    h = np.zeros(max(delay_samples)+1)
    h[0] = 1 #original signal (delta[n])
    i = 0
    #Putting the proper amplitude decay in the proper spot
    for delay in delay_samples:
        h[delay] = amplitudes[i] 
        i+=1

    #Convolve
    y_with_echo = convolve(y, h, mode='full')

    #Normalize to avoid clipping
    max_val = np.max(np.abs(y_with_echo))
    if max_val > 1.0:
         y_with_echo = y_with_echo / max_val


    # y_with_echo = y_with_echo.astype(np.int16)
    y_with_echo = (y_with_echo*32767).astype(np.int16)

    #Plotting
    #points_original = np.arange(len(y))/sr
    #points = np.arange(len(y_with_echo))/sr
    #plt.figure(figsize=(10, 6))

    #plt.subplot(2, 1, 1)
    #plt.plot(points_original, y, label="Original Audio")
    #plt.title("Original Audio Waveform")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Amplitude")
    #plt.grid(True)

    #plt.subplot(2, 1, 2)
    #plt.plot(points, y_with_echo, label="Audio with Echo", color='magenta')
    #plt.title("Audio with Echo Effect")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Amplitude")
    #plt.grid(True)

    #plt.tight_layout()
    #plt.show()

    return y_with_echo, sr

if __name__ == "__main__":
    y, sr = librosa.load("trumpet.wav")
    y,sr = echo(y,sr)
    sf.write("output.wav", y, sr)