import librosa
import numpy as np
import scipy
import pitch
asd
def scale_autotune(y, sr):

    # Major and Minor Scales used for determing the notes in the scale (0 = root, etc)
    def major_scale(root):
        return sorted([(root + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]])

    def minor_scale(root):
        return sorted([(root + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10]])
    
    # Butterworth Lowpass
    def butter_lowpass(cutoff_hz, sr, order):
        normalized_cutoff = cutoff_hz / sr
        b, a = scipy.signal.butter(order, normalized_cutoff, btype='low', analog=False)
        return b, a

    # Pitch Shifting
    def pitch_shift(y_frame, sr, n_steps, hop_length=512, n_fft=2048):

        # Calculates the STFT of the frame -> Does vocoder shift -> Calculates ISTFT (Similar framework to pitch shift effect)
        D = librosa.stft(y_frame, n_fft=n_fft, hop_length=hop_length)
        rate = 2 ** (-n_steps / 12)
        D_shifted = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
        y_shifted = librosa.istft(D_shifted, hop_length=hop_length, length=len(y_frame))

        return y_shifted

    # Key detection
    def detect_key(y_segment, sr):

        if len(y_segment) < 1024 or np.max(np.abs(y_segment)) < 1e-5:
            return 'major', 0
        
        # Determines the amount of energy of each pitch in 12 bins to denote C, C#, etc. and determines mean
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=1024, tuning=0)
        mean_chroma = np.mean(chroma, axis=1)

        # Template major and minor
        template_M = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        template_m = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        scores = []

        # Computes the dot product of the templates and the mean_chroma vector
        # Higher the value = more likely the scale to be -> Correlation of energy to appropriate scale
        for i in range(12):
            score_M = np.dot(np.roll(template_M, i), mean_chroma)
            score_m = np.dot(np.roll(template_m, i), mean_chroma)
            scores.append(('major', i, score_M))
            scores.append(('minor', i, score_m))

        # Finds the highest relevant score and declares that to be the segments scale (Returns the major or minor and its base note)
        key_type, key_root, _ = max(scores, key=lambda x: x[2])
        return key_type, key_root


    #------------------ MAIN CODE ----------------------#

    # Setup parameters for segment duration and necessary frame size / hop length
    audio_length = 2.5
    hop_length = 512
    frame_length = 2048
    
    # Low pass to get rid of high pitch noise
    b, a = butter_lowpass(cutoff_hz=18000, sr=sr, order=5)
    y = scipy.signal.lfilter(b, a, y)

    # Double checking minimum size requiremnts / not enough energy
    if np.max(np.abs(y)) < 1e-5 or len(y) < 2048:
        f_0 = np.full(len(y) // hop_length, np.nan)
    else:
        # Storing fundamental pitches of each frame
        f_0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )

    #-------Storage of correct pitches---------*
    f0_corrected = np.copy(f_0)

    # Calculating the number of frames per segment and the num of total segments
    frames_per_seg = int(audio_length * sr / hop_length)
    num_seg = int(np.ceil(len(f_0) / frames_per_seg))

    for i in range(num_seg):
        start = i * frames_per_seg
        end = min((i + 1) * frames_per_seg, len(f_0))
        y_seg = y[int(start * hop_length):int(end * hop_length)]

        # Sizing check of segment
        if len(y_seg) < 1024 or np.max(np.abs(y_seg)) < 1e-5:
            continue 

        # Detecting key of the segment
        key, root = detect_key(y_seg, sr)
        scale = major_scale(root) if key == 'major' else minor_scale(root)

        # Scaling through each frame of the segment
        for j in range(start, end):
            if not np.isnan(f_0[j]):

                # Converts the fundamental frequency to the MIDI scale and rounds to best fit MIDI
                midi_converted = librosa.hz_to_midi(f_0[j])
                rounded = int(np.round(midi_converted))
                base_note = rounded % 12

                # Calculating distance of fundamental note to the allowed notes per the scale and finds shortest distance
                # Base = 1 and scale [0, 3, 5...], it'll assume 0 to be the closest
                delta = [(abs((base_note - n + 12) % 12), n) for n in scale]
                _, closest = min(delta)
                shift = (closest - base_note + 12) % 12

                # Distance will dictate if it is a round down or a round up and store accordingly
                corrected_midi = rounded + shift if shift < 6 else rounded - (12 - shift)
                f0_corrected[j] = librosa.midi_to_hz(corrected_midi)

    #--------Pitch Shift and Output section----------#
    y_autotuned = np.zeros_like(y)

    for i, (f_orig, f_corr) in enumerate(zip(f_0, f0_corrected)):
        # Going through each frame
        start_sample = i * hop_length
        end_sample = start_sample + frame_length

        # Overextending
        if end_sample > len(y):
            break 
        
        frame = y[start_sample:end_sample]

        # Condition checks
        if (
            not np.isnan(f_orig) and not np.isnan(f_corr) and f_orig > 0 and len(frame) >= 1024 and np.max(np.abs(frame)) > 1e-5
        ):
            # Figures out required semitone shift and utilizes pitch_shift function in autotune file
            n_steps = 12 * np.log2(f_corr / f_orig)
            shifted = pitch_shift(frame, sr=sr, n_steps=n_steps, hop_length=hop_length)
            y_autotuned[start_sample:end_sample] += shifted[:len(frame)]
        else:
            y_autotuned[start_sample:end_sample] += frame

    # Normalizing for return in proper format
    y_tuned_normalized = y_autotuned / np.max(np.abs(y_autotuned))
    y_tuned_int16 = np.int16(y_tuned_normalized * 32767)
    return y_tuned_int16, sr