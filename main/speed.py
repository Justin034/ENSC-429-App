from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
import librosa

# compute STFT and return complex spectrogram (magnitue + j * phase)
# x = audio waveform
# frame size = length of each frame
# hop size = how far to move along when taking new frame
    # hop size < frame size = overlapping frames
def manualStft(x, frameSize=2048, hopSize=512):
    # create hanning window
    window = np.hanning(frameSize)
    # store FFT of frames in list
    frames = []
    # loop through signal in intervals of hop size
    for i in range(0, len(x) - frameSize, hopSize):
        # apply window to frame
        frame = x[i:i+frameSize] * window
        # compute fast fourier transform of frame
        spectrum = fft(frame)
        # store frequency of frame in list
        frames.append(spectrum)
    # return array of frequency spectrum
    # rows = frequencies
    # columns = time
    return np.array(frames).T  

# compute inverse of STFT and return reconstructed audio in time domain
def manaulIstft(X, frameSize=2048, hopSize=512):
    # create hanning window
    window = np.hanning(frameSize)
    # get number of frames 
    numFrames = X.shape[1]
    # initalize empty output signal
    output = np.zeros(frameSize + hopSize * (numFrames - 1))
    # loop through each frame to reconstruct time domain signal
    for i in range(numFrames):
        # convert frequency back to time
        frame = np.real(ifft(X[:, i]))
        # add frames to output signal
        output[i * hopSize : i * hopSize + frameSize] += frame * window
    return output

# phase vocoder algorithim for time shifting with pitch preserved
def manualPhaseVocoder(x, stretch, frameSize=2048, hopSize=512):
    # get STFT of input signal
    X = manualStft(x, frameSize, hopSize)
    # unpack dimensions of array to get number of frequency bins and frames
    numBins, numFrames = X.shape

    # frame position to use when reconstructing stretched/compressed audio
    # arrange function creates array of evenly spaced numbers by increment stretch
    framePos = np.arange(0, numFrames, stretch)
    # create empty spectrogram matrix for 64 bit complex number
    outputSpectrogram = np.zeros((numBins, len(framePos)), dtype=np.complex64)
    # initalize with phase of first frame
    phaseAccumulator = np.angle(X[:, 0])

    # loop through stretched/compressed time postions
    for i, t in enumerate(framePos):
        # integer portion of frame index
        tInt = int(np.floor(t))
        # break out of loop if exceed number of frames
        if tInt + 1 >= numFrames:
            break
        # fraction portion of frame postion 
        fraction = t - tInt

        # interpolate magnitude between two frames
        mag = (1 - fraction) * np.abs(X[:, tInt]) + fraction * np.abs(X[:, tInt + 1])
        # calculate phase difference between two consecutive frames
        phaseDiff = np.unwrap(np.angle(X[:, tInt + 1]) - np.angle(X[:, tInt]))
        # update phase
        phaseAccumulator += phaseDiff
        # rebuild frame with updated magnitude and phase
        outputSpectrogram[:, i] = mag * np.exp(1j * phaseAccumulator)

    # convert and output stretched/compressed audio in time domain 
    return manaulIstft(outputSpectrogram, frameSize, hopSize)

# speed filter
# change playback speed without altering pitch
# y = audio signal
# sr = sample rate
def speeder(y, sr, stretchFactor = 1.0):

    # convert to mono if audio is stereo 
    if y.ndim == 2:
        y = y.mean(axis=1)

    # normalize input so amplitude between 1/-1 and convert to type float32 for precise calculations
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    stretched = manualPhaseVocoder(y, stretch=stretchFactor)

    # normalize output to avoid clipping
    stretched /= np.max(np.abs(stretched))

    return (stretched * 32767).astype(np.int16), sr
