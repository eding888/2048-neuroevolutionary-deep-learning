import time
import subprocess

hours_to_record = 24  # Number of hours you want to keep this running
interval = 600       # One hour interval in seconds
record_duration = 2  # Record for 60 seconds

for _ in range(hours_to_record):
    # Command to start recording. Adjust the screen size and output file as needed.
    command = [
        'ffmpeg',
        '-f', 'avfoundation',  # Change 'x11grab' to 'gdigrab' on Windows or 'avfoundation' on Mac.
        '-i', '1:0',  # Change this depending on your system, ':0.0' is typical for Linux
        '-t', '00:01:00',
        '-r', '30',  # Duration of recording in seconds
        f'output_{int(time.time())}.mp4'  # Output file name with timestamp
    ]
    # Start recording
    subprocess.run(command)
    # Wait for the next hour minus the recording time
    time.sleep(interval - record_duration)