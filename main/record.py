import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading

#Shared state stored in a dictionary
shared_state = {
    "buffer": [],
    "stop_flag": threading.Event()
}

#The record function constantly calls this
def audio_callback(indata, frames, time, status):
#keep checking if it should stop
    if shared_state["stop_flag"].is_set():
        raise sd.CallbackStop()
    #append any new data to the buffer. the buffer ends up as a list of multiple numpy arrays
    shared_state["buffer"].append(indata.copy())

#Function to just check if user cancels recording early
def wait_for_enter():
    input("Press Enter to stop recording early...\n")
    shared_state["stop_flag"].set()

def record_audio(filename="recorded_audio.wav", max_duration=10, sample_rate=44100):
    #Start listener thread
    #A separate thread for listening and checking cancellation
    listener_thread = threading.Thread(target=wait_for_enter)
    listener_thread.start()

    print("Recording...")
    #Try and get audio input from a mic
    try:
        with sd.InputStream(samplerate=sample_rate,
                            channels=1,
                            dtype='float32',
                            callback=audio_callback):
            sd.sleep(int(max_duration * 1000))  #milliseconds
            #if an early stop was requested, just pass it through
    except sd.CallbackStop:
        pass

    print("Recording finished.")

    #Combine the chunks of audio into one cohesive array
    audio_data = np.concatenate(shared_state["buffer"], axis=0)
    #Write the final file
    write(filename, sample_rate, audio_data)
    print(f"Saved as {filename}")

if __name__ == "__main__":
    record_audio()
    