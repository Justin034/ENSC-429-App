import sys
import os
import librosa
import pitch
import speed
import equalize
import echo
import autotune
import record
from scipy.io import wavfile
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QDialog, QSlider,
    QHBoxLayout, QDialogButtonBox
)
from PyQt6.QtCore import Qt

#--- Function storing pathing for effect to proper file/function---#
def process_audio_file(y, sr, task_data):
    if isinstance(task_data, tuple):
        task_name, params = task_data
    else:
        task_name = task_data
        params = ()

    print(f"Running task: {task_name}")

    # data y and the sample rate is sent to each file for 
    if task_name == "Pitch":
        a, b = pitch.pitcher(y, sr, params)
    elif task_name == "Autotune":
        a, b = autotune.scale_autotune(y, sr)
    elif task_name == "Speed Up":
        a, b = speed.speeder(y, sr, params)
    elif task_name == "Equalize":
        slider1, slider2, slider3 = params
        a, b = equalize.equalizer(y, sr, slider1, slider2, slider3)
    elif task_name == "Echo":
        a, b = echo.echo(y, sr)
    else:
        a, b = y, sr

    print("Task complete.\n")
    return a, b

#--- Slider window for equilizer customizing ---#
class SliderWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Equalizer Parameters")
        self.slider_values = [0, 0, 0]

        layout = QVBoxLayout()

        # Storing value and for equilizer and its displayed labels
        self.sliders = []
        self.value_labels = []

        for i in range(3):
            slider_label = QLabel(f"Band {i + 1} Gain:")
            layout.addWidget(slider_label)

            # Horizontal layout for slider + value label
            h_layout = QHBoxLayout()

            # slider ranges
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-10)
            slider.setMaximum(10)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, idx=i: self.update_slider_value(idx, val))

            value_label = QLabel("0") 
            value_label.setFixedWidth(30) 

            # Widgets for displaying on screen
            h_layout.addWidget(slider)
            h_layout.addWidget(value_label)

            layout.addLayout(h_layout)

            self.sliders.append(slider)
            self.value_labels.append(value_label)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def update_slider_value(self, index, value):
        self.slider_values[index] = value
        self.value_labels[index].setText(str(value))

    def get_values(self):
        return tuple(self.slider_values)
    

# === Main UI ===
class AudioQueueUI(QWidget):
    def __init__(self):
        super().__init__()

        # Setting up layout for the buttons and the necessary button click commands
        self.setWindowTitle("Audio Task Queue")
        self.task_queue = deque()
        self.file_path = None
        self.y = None
        self.sr = None

        layout = QVBoxLayout()

        self.label = QLabel("No audio selected.")
        layout.addWidget(self.label)

        self.record_button = QPushButton("Record Audio")
        self.record_button.clicked.connect(self.recording_function)
        layout.addWidget(self.record_button)

        self.select_button = QPushButton("Select Audio File")
        self.select_button.clicked.connect(self.select_audio_file)
        layout.addWidget(self.select_button)

        # Task buttons and their respective task names
        self.task_buttons = {
            "Pitch": QPushButton("Pitch Option"),
            "Autotune": QPushButton("Autotune Option"),
            "Speed Up": QPushButton("Speed Up Option"),
            "Equalize": QPushButton("Equalizer Option"),
            "Echo": QPushButton("Echo Option")
        }

        # Other necessary buttons in terms of current queue and the run all tasks buttons
        for task_name, button in self.task_buttons.items():
            button.clicked.connect(lambda _, t=task_name: self.add_task(t))
            layout.addWidget(button)

        self.confirm_button = QPushButton("Show Current Task Queue")
        self.confirm_button.clicked.connect(self.show_queue)
        layout.addWidget(self.confirm_button)

        self.run_button = QPushButton("Run All Tasks")
        self.run_button.clicked.connect(self.run_tasks)
        layout.addWidget(self.run_button)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)

        self.setLayout(layout)

    # Goes to recording function
    def recording_function(self):
        record.record_audio(filename="recorded_audio.wav", max_duration=10, sample_rate=44100)

    # Once the file is loaded, it sets the necessary data and sample rate parameters for other functions
    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)"
        )
        if file_path:
            try:
                # Sets up proper parameters for the data and sample rate values to be used
                self.y, self.sr = librosa.load(file_path, sr=None)
                self.file_path = file_path
                self.label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.text_output.append(f"Loaded audio: {os.path.basename(file_path)} (sr = {self.sr})")
                self.task_queue.clear()
                self.show_queue()
            except Exception as e:
                self.text_output.append(f"Failed to load audio: {e}")
                self.file_path = None
                self.y = None
                self.sr = None

    # Used for adding the task into a queue, allowing for the organization of the effects
    def add_task(self, task_name):
        if self.y is None:
            self.text_output.append("Please select a valid audio file first.")
            return

        # Setting up necessary windows for equilizing, speed and pitch parameters
        if task_name == "Equalize":
            dialog = SliderWindow(self)
            if dialog.exec():
                values = dialog.get_values()
                self.task_queue.append((task_name, values))
                self.text_output.append(f"Task added: {task_name} with values {values}")
        
        # Parameters for speeding up 
        elif task_name == "Speed Up":
            from PyQt6.QtWidgets import QInputDialog
            speed_value, ok = QInputDialog.getDouble(
                self,
                "Adjust Speed",
                "Enter speed factor (e.g., 0.5 for slower, 1.0 for normal, 2.0 for faster):",
                1.0,  # Default value
                0.1,  # Min
                5.0,  # Max
                2     # Decimal places
            )
            if ok:
                self.task_queue.append((task_name, speed_value))
                self.text_output.append(f"Task added: {task_name} with speed {speed_value}x")

        # Parameters values for pitch set up
        elif task_name == "Pitch":
            from PyQt6.QtWidgets import QInputDialog
            semitone_value, ok = QInputDialog.getInt(
                self,
                "Adjust Pitch",
                "Enter pitch shift (in semitones, e.g., -12 for down, 0 for no change, +12 for up):",
                0,     # Default value (neutral)
                -24,   # Minimum semitone shift (2 octaves down)
                24,    # Maximum semitone shift (2 octaves up)
                1      # Step size
            )
            if ok:
                self.task_queue.append((task_name, semitone_value))
                self.text_output.append(f"Task added: {task_name} with pitch shift {semitone_value} semitones")

        else:
            self.task_queue.append(task_name)
            self.text_output.append(f"Task added: {task_name}")

        self.show_queue()

    # Returns the current queue of tasks
    def show_queue(self):
        self.text_output.append("Current Queue:")
        if not self.task_queue:
            self.text_output.append("Task queue is empty.")
            return

        for i, task in enumerate(self.task_queue, 1):
            if isinstance(task, tuple):
                name, params = task
                self.text_output.append(f"{i}. {name} {params}")
            else:
                self.text_output.append(f"{i}. {task}")

    # Once pressed, will check for proper file setup and if passed, it will pop from the queue and apply the effect
    def run_tasks(self):
        if self.y is None or self.sr is None:
            self.text_output.append("No valid audio loaded.")
            return

        if not self.task_queue:
            self.text_output.append("No tasks to run.")
            return

        self.text_output.append("Running Tasks...\n")

        while self.task_queue:
            task = self.task_queue.popleft()
            self.text_output.append(f"Running: {task}")
            self.y, self.sr = process_audio_file(self.y, self.sr, task)

        self.text_output.append("All tasks completed.\n")
        wavfile.write("Edited_File.wav", self.sr, self.y)
        self.show_queue()


#--- Main Entry ---#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioQueueUI()
    window.resize(400, 500)
    window.show()
    sys.exit(app.exec())
