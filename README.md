
# FSD Stop Detection Dashcam Video Editor

Auto detection of stop / start in driving videos. This application is designed to help users process dashcam footage where the speedometer is visible and automatically detect "boring parts" of the video—those where the car is stationary, such as at stoplights. 

By leveraging Optical Character Recognition (OCR) and frame processing, the application identifies when the car stops and moves, logs these events, and can export clips of only the moving segments for easier review.

https://github.com/user-attachments/assets/1c735f70-c1e7-476f-b3a4-356a4186865a

## Features

- **Detect and log stop/start events**: Automatically identify when a vehicle is stationary and moving based on the visible speedometer.
- **Generate logs**: Create a detailed log with timecodes showing when the vehicle starts and stops.
- **Create an annotated video**: Optionally overlay the detected speed and vehicle status (Stopped/Moving) on the video.
- **Export clips of moving segments**: Automatically cut out and export only the parts of the video when the car is moving.
- **Customizable**: Offers multiple configuration options, including the ability to disable video generation or specify the output directory for clips.

## Web Version Here:

http://aiblocks.ai/users/bdekraker/fsd-detect/

## Requirements

To use this application, you will need to install a few dependencies:

1. **Python 3.x**: Ensure you have Python 3 installed.
2. **Tesseract OCR**: Required for reading the speedometer in the dashcam footage.
3. **FFmpeg**: Required for frame extraction, video processing, and merging.

   This version expects a digital speedometer to be visible in the video, in order to properly detect when the vehicle is stopped or moving.

### Python Libraries

The following Python libraries are required and can be installed using `pip`:

- `opencv-python`
- `pytesseract`
- `matplotlib`
- `Pillow`
- `numpy`

You can install the dependencies all at once by using the `requirements.txt` file included in this repository. To install the dependencies with pip, run the following command:

```bash
pip install -r requirements.txt
```

This will automatically install all the necessary Python packages.

## Installation

### Step 1: Install Python 3.x

Make sure you have Python 3.x installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Step 2: Install Tesseract OCR

#### Windows
1. Download Tesseract from the official GitHub repo: [Tesseract OCR for Windows](https://github.com/tesseract-ocr/tesseract).
2. Install it.
3. After installation, make sure to add the path to Tesseract to your system's environment variables. For example:
   ```plaintext
   C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

#### Linux
You can install Tesseract using your package manager:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### macOS
1. Install Homebrew if you haven't already. Homebrew is a package manager for macOS and is the easiest way to install Tesseract. To install Homebrew, open your terminal and run:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. After Homebrew is installed, install Tesseract by running the following command in your terminal:
   ```bash
   brew install tesseract
   ```

3. Verify that Tesseract was installed correctly by running:
   ```bash
   tesseract --version
   ```

4. You should see version information about Tesseract, which confirms it is installed and available in your system's PATH.

5. If you installed Tesseract using Homebrew, the default location for Tesseract on macOS is `/usr/local/bin/tesseract`. The application should automatically detect it, but if needed, you can specify the path manually in your code:
   ```python
   pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
   ```

---

### Step 3: Install FFmpeg

#### Windows
1. Download the FFmpeg executable from [ffmpeg.org](https://ffmpeg.org/download.html).
2. Follow the instructions for adding FFmpeg to your PATH.

#### Linux
Install FFmpeg using your package manager:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
1. Similar to Tesseract, the easiest way to install FFmpeg on macOS is using Homebrew. If you haven't installed Homebrew, follow the instructions in the **macOS** section under "Install Tesseract OCR."

2. To install FFmpeg with Homebrew, run the following command in your terminal:
   ```bash
   brew install ffmpeg
   ```

3. Verify the installation by running:
   ```bash
   ffmpeg -version
   ```

4. You should see version information about FFmpeg, which confirms it is installed and available in your system's PATH.


### Step 4: Install Required Python Libraries

Once Python is installed, you can install the required dependencies by running the following command:
```bash
pip install opencv-python pytesseract matplotlib Pillow numpy
```

Alternatively, you can install all the required Python libraries from the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run the Application

You can run the application by executing the Python script:

```bash
python speed.py
```

This will open a file dialog where you can select your dashcam video file (formats like `.mp4`, `.avi`, `.mov`, `.mkv` are supported).

### Step 2: Draw the Bounding Box



https://github.com/user-attachments/assets/cdb87143-26f1-4616-bef4-3b4bca789db6



The application will display the first frame of the video. Draw a bounding box around the area of the speedometer. Once you have done this, press `Enter` to confirm.



### Step 3: Output Files

![image](https://github.com/user-attachments/assets/25f2fe0f-84f2-4a04-9b26-6c13e848cc4f)


- **Dwell log**: A log file `dwell_log.txt` will be generated, recording the timestamps of when the car starts and stops moving.
- **Annotated video** (optional): The script can generate a new video file (`output_video.mp4`) with the speed and status (Moving/Stopped) overlaid on the frames.

## Optional Flags

You can control various aspects of the application using command-line flags.

### `--fps`

Specify an FPS for processing of the video. This affects both reading the speed and outputting an annotated (speed overlayed) video. Default FPS is 3. Higher FPS takes longer to process.

### `--no-video`

By default, the script generates an annotated output video. You can disable this feature by using the `--no-video` flag, which will only generate the log file:

```bash
python speed.py --no-video
```

### `--export-clips`

This option enables the automatic exporting of moving segments from the video, so you only retain the parts where the car is moving. The moving segments are saved as separate video clips.

```bash
python speed.py --export-clips
```

### `--clip-folder`

You can specify a custom folder for saving the clips generated by the `--export-clips` flag:

```bash
python speed.py --export-clips --clip-folder path/to/clip_folder
```

If no folder is specified, clips will be saved in a default folder named `clips`.

## Example Workflow

1. **Basic Run**: Run the script, generate both the log and the annotated video.
   ```bash
   python speed.py
   ```

2. **Generate Logs Only (no video output)**:
   ```bash
   python speed.py --no-video
   ```

3. **Export Moving Segments**: Split the video into multiple clips, with each clip containing only the time when the car is moving.
   ```bash
   python speed.py --export-clips
   ```

4. **Export Moving Segments to a Custom Folder**:
   ```bash
   python speed.py --export-clips --clip-folder path/to/clip_folder
   ```

## Cleanup

After processing is completed, temporary frames that were extracted from the video will be cleaned up automatically to save disk space.

## Troubleshooting

### Tesseract Not Found

If the script fails with a message saying "Tesseract not found", ensure that:

1. Tesseract is installed.
2. You have added the path to `tesseract.exe` (on Windows) to your system’s PATH environment variable.

### FFmpeg Not Found

If FFmpeg is not found, install it and ensure it is added to your system’s PATH variable. You can check if FFmpeg is installed correctly by running the following in your terminal:

```bash
ffmpeg -version
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

If you'd like to contribute to this project, feel free to submit issues and pull requests. Be sure to follow best practices and make sure all features are tested before submission.

## Acknowledgments

- **Tesseract OCR** for text extraction.
- **FFmpeg** for video processing.
- The **OpenCV** and **Matplotlib** communities for their invaluable Python libraries.
