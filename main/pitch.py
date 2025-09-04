import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import resample

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
def manualIstft(X, frameSize=2048, hopSize=512):
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

# Phase vocoder algorithim for time stretching without changing pitch
# used as a step in pitch shifting
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
    return manualIstft(outputSpectrogram, frameSize, hopSize)

def manualResample(signal, targetLength):
    """
    Resample an audio signal to a new length using linear interpolation.

    Args:
        signal (np.ndarray): Input audio signal (1D array)
        targetLength (int): Desired number of samples in output signal

    Returns:
        np.ndarray: Resampled audio signal
    """
    originalLength = len(signal)

    # Generate new sample positions (target grid) mapped to original positions
    newPositions = np.linspace(0, originalLength - 1, targetLength)

    # Get indices of nearest lower integer sample points
    leftIndices = np.floor(newPositions).astype(int)
    rightIndices = np.clip(leftIndices + 1, 0, originalLength - 1)  # avoid overflow

    # Linear interpolation weights
    weights = newPositions - leftIndices

    # Perform interpolation: value = (1-w)*left + w*right
    resampledSignal = (1 - weights) * signal[leftIndices] + weights * signal[rightIndices]

    return resampledSignal.astype(np.float32)

# pitch shifting using time stretching and resampling
def pitchShift(x, sr, semitones):
    # convert semitones to pitch ratio (12 semitones = 1 octave = 2x frequency)
    ratio = 2 ** (semitones / 12.0)

    # time stretch audio inversely to pitch ration
    stretched = manualPhaseVocoder(x, stretch=1/ratio)

    # resample to original length
    resampled = manualResample(stretched, len(x))
    return resampled

# pitch shifting function to be called
def pitcher(y, sample, semitones = 0):

    sr = sample
    data = y

    # convert to float 32 type and mono
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)

    # normalize input
    data /= np.max(np.abs(data))

    # apply pitch shifting up to 5 semitones
    shifted = pitchShift(data, sr, semitones = semitones)

    # normalize to avoid clipping and convert back to 16 bit
    shifted /= np.max(np.abs(shifted))
    return (shifted * 32767).astype(np.int16), sr 