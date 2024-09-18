import os
import sys
import cv2
import pytesseract
import subprocess
import re
import argparse
from tkinter import Tk, filedialog
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
import numpy as np

# Ensure Tesseract is installed and configured
# On Windows, set the tesseract_cmd to the installed location
# On Linux, Tesseract is usually in the PATH
if os.name == 'nt':  # If OS is Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

# Global variables
last_valid_speed = "0"       # Store last valid speed detected
dwell_start_time = None      # Time when speed first hit 0
is_stopped = False           # Whether the car is stopped
dwell_threshold = 5          # Threshold in seconds to detect stop/start dwell
bbox = None                  # Global variable to store the bounding box coordinates

# New variables for exporting clips
moving_segments = []         # List to store (start_time, end_time) tuples when the car is moving

def get_original_fps(video_path):
    """
    Extract the original FPS from the video using ffmpeg.
    """
    command = ['ffmpeg', '-i', video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Combine stdout and stderr to catch FPS info
    output = result.stdout + result.stderr
    match = re.search(r'(\d+(?:\.\d+)?) fps', output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Unable to extract FPS from video")

def extract_frames(video_path, output_dir, fps=3):
    """
    Extract frames from the video using ffmpeg at the specified FPS.
    """
    print(f"Extracting frames from video at {fps} FPS...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = [
        'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', os.path.join(output_dir, 'frame_%04d.png')
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"Frames extracted to {output_dir}.")

def line_select_callback(eclick, erelease):
    """
    Callback function for RectangleSelector to record the bounding box coordinates.
    """
    global bbox
    bbox = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
    print(f"Bounding box selected: {bbox}")

def on_key_press(event):
    """
    Event handler for key press events in the matplotlib window.
    Closes the window when 'Enter' is pressed.
    """
    if event.key == 'enter':
        print("Bounding box confirmed. Processing frames...")
        plt.close()

def draw_bounding_box(image):
    """
    Displays an image and allows the user to draw a bounding box over the speedometer.
    """
    global bbox
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display

    # Create the RectangleSelector and key event for submitting
    rect_selector = RectangleSelector(
        ax, line_select_callback,
        interactive=True, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels'
    )

    # Connect the "enter" key press to submit the bounding box
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.show()

    return bbox

def read_speed(frame, bbox):
    """
    Reads the speed from the given frame using OCR within the bounding box.
    """
    global last_valid_speed
    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within the frame dimensions
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Limiting Tesseract OCR to recognize only numerical digits
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

    # Perform OCR
    text = pytesseract.image_to_string(gray_roi, config=custom_config).strip()

    # If OCR result is blank, use the last valid speed
    if not text:
        print(f"Blank or null detection. Using last detected speed: {last_valid_speed}")
        return last_valid_speed

    # Update last valid speed
    print(f"Detected speed: {text}")
    last_valid_speed = text
    return text

def frame_to_timecode(processed_frame_index, original_fps, processing_fps):
    """
    Converts processed frame index to timecode based on original video FPS.
    """
    # Calculate the corresponding frame in the original video
    original_frame_index = processed_frame_index * (original_fps / processing_fps)

    # Convert the original frame index into seconds
    seconds = original_frame_index / original_fps
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"

def detect_zero_speed_dwell(current_speed, original_fps, processing_fps, frame_index, log_file):
    """
    Detects when the vehicle stops and starts moving, logs the events with timecodes.
    """
    global dwell_start_time, is_stopped, moving_segments
    current_time = frame_to_timecode(frame_index, original_fps, processing_fps)  # Adjusted timecode calculation

    if current_speed == "0":
        if dwell_start_time is None:  # Car just stopped
            dwell_start_time = current_time
            if not is_stopped:
                # Car has been moving, now stopped. Record the end of the moving segment.
                if moving_segments:
                    moving_segments[-1] = (moving_segments[-1][0], current_time)
        elif not is_stopped:
            # Car has been stopped, log the event
            log_file.write(f"Stopped at {dwell_start_time}\n")
            print(f"Stopped at {dwell_start_time}")
            is_stopped = True
    else:
        if is_stopped:
            # Car starts moving again, log the event
            log_file.write(f"Started moving at {current_time}\n\n")  # Add newline after "Started moving"
            print(f"Started moving at {current_time}")
            is_stopped = False
            dwell_start_time = None  # Reset dwell start time
            # Start a new moving segment
            moving_segments.append((current_time, None))
        elif not moving_segments:
            # Car has been moving since the beginning
            moving_segments.append((current_time, None))

def overlay_speed_status(frame, speed, is_stopped):
    """
    Overlays the detected speed and status (Stopped/Moving) on the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = "Stopped" if is_stopped else "Moving"
    cv2.putText(frame, f'Speed: {speed}', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Status: {status_text}', (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def merge_frames_to_video(output_dir, output_video_path, fps=3):
    """
    Re-merges frames into a video using ffmpeg.
    """
    print(f"Merging frames into a video...")
    command = [
        'ffmpeg', '-r', str(fps), '-i', os.path.join(output_dir, 'frame_%04d_overlay.png'),
        '-vcodec', 'libx264', '-y', output_video_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"Video created: {output_video_path}")

def export_moving_segments(video_path, segments, clip_folder):
    """
    Exports the moving segments as separate video clips using ffmpeg.
    """
    if not os.path.exists(clip_folder):
        os.makedirs(clip_folder)
    print(f"Exporting moving segments to {clip_folder}...")
    for idx, (start_time, end_time) in enumerate(segments):
        if end_time is None:
            # If the last segment has no end_time, set it to the video's duration
            end_time = get_video_duration(video_path)
        output_clip = os.path.join(clip_folder, f"clip_{idx+1:03}.mp4")
        command = [
            'ffmpeg', '-i', video_path, '-ss', start_time, '-to', end_time, '-c', 'copy', output_clip
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Exported clip: {output_clip}")

def get_video_duration(video_path):
    """
    Gets the total duration of the video in HH:MM:SS.mmm format.
    """
    command = ['ffmpeg', '-i', video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output = result.stdout + result.stderr
    match = re.search(r'Duration: (\d+:\d+:\d+\.\d+)', output)
    if match:
        return match.group(1)
    else:
        raise ValueError("Unable to extract video duration")

def cleanup_frames(output_dir):
    """
    Deletes the temporary frames directory and its contents.
    """
    if os.path.exists(output_dir):
        print(f"Cleaning up temporary frames in {output_dir}...")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        os.rmdir(output_dir)
        print(f"Temporary frames cleaned up.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a video to detect stop/start events based on speedometer readings.")
    parser.add_argument('--no-video', action='store_true', help="Disable creation of the output video.")
    parser.add_argument('--export-clips', action='store_true', help="Export moving segments as separate video clips.")
    parser.add_argument('--clip-folder', type=str, default='clips', help="Folder to save the exported clips.")
    args = parser.parse_args()

    # Get video file from user
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not video_path:
        print("No file selected.")
        return

    print(f"Selected video: {video_path}")

    # Check if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("ffmpeg is not installed or not found in PATH.")
        sys.exit(1)

    # Extract the original FPS from the input video
    original_fps = get_original_fps(video_path)
    print(f"Original FPS of the video: {original_fps}")

    # Extract frames from the video
    output_dir = 'temp_frames'
    extract_frames(video_path, output_dir, fps=3)

    # Get the first frame to allow user to draw a bounding box
    first_frame_path = os.path.join(output_dir, 'frame_0001.png')
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print("No frames extracted.")
        cleanup_frames(output_dir)
        return

    # Show the frame and get the bounding box from the user
    print("Displaying the first frame. Draw a bounding box over the speedometer and press Enter.")
    bbox = draw_bounding_box(first_frame)

    # Open log file to record stop/start events
    log_file_path = 'dwell_log.txt'
    with open(log_file_path, 'w') as log_file:
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png') and 'overlay' not in f])
        print(f"Processing {len(frame_files)} frames...")

        for frame_index, frame_file in enumerate(frame_files):
            frame_path = os.path.join(output_dir, frame_file)
            frame = cv2.imread(frame_path)
            speed = read_speed(frame, bbox)

            # Detect zero speed dwell and log stop/start events using original FPS
            detect_zero_speed_dwell(speed, original_fps, 3, frame_index, log_file)

            # Overlay speed and status on the frame
            if not args.no_video:
                overlayed_frame = overlay_speed_status(frame, speed, is_stopped)
                overlay_frame_path = os.path.join(output_dir, frame_file.replace('.png', '_overlay.png'))
                cv2.imwrite(overlay_frame_path, overlayed_frame)
            print(f"Processed frame {frame_file}")

    # Re-merge frames into a new video if not disabled
    if not args.no_video:
        output_video_path = 'output_video.mp4'
        merge_frames_to_video(output_dir, output_video_path)
        print(f"Output video created: {output_video_path}")

    print(f"Dwell log saved to {log_file_path}")

    # Export moving segments as clips if enabled
    if args.export_clips:
        if moving_segments:
            print("Exporting moving segments as video clips...")
            export_moving_segments(video_path, moving_segments, args.clip_folder)
        else:
            print("No moving segments detected to export.")

    # Clean up temporary frames
    cleanup_frames(output_dir)

if __name__ == "__main__":
    main()
