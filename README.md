# YOLO Train and Detect GUI

This is a desktop GUI app for training YOLO models such as YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, and YOLOv8, then running inference on images, videos, or a camera from the same application.

The app is built with CustomTkinter, PyTorch, Ultralytics, OpenCV, and Pillow.

<img width="2362" height="740" alt="スクリーンショット 2026-05-02 145213" src="https://github.com/user-attachments/assets/01207c37-0a9d-4a3d-a2ea-6c36053befa0" />

https://youtu.be/Jk-JkBn4Na0?si=3qG5Ev82_yoJZp7x

## Setup

Use a virtual environment so the app dependencies do not conflict with other Python projects.

### Windows

Open PowerShell in the project folder.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

Open a terminal in the project folder.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start The App

Run this from the activated virtual environment:

```bash
python main.py
```

The app opens maximized. The sidebar lets you switch between training, image/video inference, and camera inference. It also has language, display mode, and UI scale settings.

## Preparing Training Data

The `Train` window expects YOLO-format training data. This is the data used to train your own object detection model.

At minimum, each image must have a label text file with the same file name stem:

```text
image_001.jpg
image_001.txt
image_002.jpg
image_002.txt
```

Each line in a label file must be:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Important notes:

- `class_id` starts from `0`.
- Coordinates must be normalized between `0.0` and `1.0`.
- The class order in the GUI must match the label IDs.
- If `0` means `duck` in your label files, the first class name in the GUI must be `duck`.
- Do not put extra blank class names in the class name box.

Example for two classes:

```text
duck
chicken
```

In this example:

- `class 0` is `duck`
- `class 1` is `chicken`

If your selected folder already contains this structure, the app uses it:

```text
train/images
train/labels
val/images
val/labels
```

If not, the app creates a temporary 80/20 train/validation split from matching image and `.txt` pairs.

## Train Window

Use the `Train` window when you want to train a new YOLO model from your own labeled images.

Fill in each field as follows:

- `Project name`: A short experiment name. Use only letters, numbers, hyphens, and underscores. Example: `duck_detection_001`.
- `Training data folder`: The folder containing your YOLO-format images and label files.
- `Training output folder`: The folder where trained weights and results should be saved.
- `YOLO model`: The base model to start from, such as `YOLO26-Nano` or `YOLO11-Medium`.
- `Input size`: Image size used during training. A common value is `640`.
- `Epochs`: Number of training epochs. Example: `100`.
- `Batch size`: Number of images per batch. Example: `16`. If training runs out of GPU memory, lower this value.
- `Class names`: One class name per line. The left side shows `class 0`, `class 1`, and so on to help match label IDs.

Before starting, check the device panel. If it says GPU is available, training should use CUDA. If it says CPU only, training will be much slower.

Click `Start Training` to begin. While training is running:

- the start button changes to a stop button
- the progress bar below the button moves
- logs appear on the right
- ETA is shown after epoch progress is available
- unrelated controls are locked to prevent accidental setting changes

## Training Outputs

Training results are copied to:

```text
<selected_output_folder>/<project_name>/
```

The most important files are:

```text
weights/best.pt
weights/last.pt
```

The app also writes:

```text
training_environment.txt
```

This file records useful experiment information, including Python library versions, PyTorch CUDA status, CUDA/cuDNN information, GPU name, VRAM, and NVIDIA driver information when available.

## Inference (Image/Video) Window

Use `Inference (Image/Video)` when you want to run object detection on existing image or video files.

Fill in each field as follows:

- `Image/video folder`: A folder containing images, videos, or both.
- `Inference model (.pt)`: A trained YOLO weight file. Usually this is `weights/best.pt`.
- `Confidence`: Detection confidence threshold from `0.01` to `1.0`. The default is `0.5`.

Lower confidence values detect more objects but may increase false detections. Higher values show only more confident detections.

Click `Start Detection` to run inference. Results are saved under:

```text
<selected_media_folder>/results/<timestamp>/
```

After inference finishes, use the previous/next buttons or left/right arrow keys to review result images.

## Inference (Camera) Window

Use `Inference (Camera)` when you want to run real-time detection from a camera connected to the PC.

Fill in each field as follows:

- `Inference model (.pt)`: A trained YOLO weight file, usually `weights/best.pt`.
- `Save folder`: The folder where captured images and detection text files should be saved.
- `Camera device`: Select the camera to use. Press `Refresh` if you connected a camera after opening the window.
- `Confidence`: Detection confidence threshold from `0.01` to `1.0`. The default is `0.5`.

The app searches available OpenCV camera devices. On Windows, it also tries to show camera names when the OS exposes them. If a name is not available, the camera may appear as `Camera 0`, `Camera 1`, and so on.

Click `START` to begin camera inference. While it is running:

- `STOP` stops camera inference
- `Enter` saves the current frame
- `Esc` stops camera inference
- unrelated controls are locked

Captured files are saved as:

```text
<timestamp>_<scene_id>_origin.png
<timestamp>_<scene_id>_detection.jpg
<timestamp>_<scene_id>_detection.txt
```

## Sidebar Controls

The sidebar contains app-level controls:

- `Language`: Switches between Japanese and English.
- `Appearance`: Switches dark/light display mode.
- `Scale`: Changes the UI size percentage. Use `100` for normal size.

These controls are disabled while training or inference is running. This prevents the active screen from being rebuilt while a job is in progress.

## Troubleshooting

### GPU Is Not Used

Check the device panel in the `Train` window before starting. If CUDA is not detected, the app will train on CPU.

Common causes:

- NVIDIA driver is missing or too old.
- The installed PyTorch package is CPU-only.
- The virtual environment is not the one you expected.
- The GPU is disabled or unavailable to Python.

### Training Fails Immediately

Check these points:

- Project name uses only letters, numbers, hyphen, and underscore.
- The training data folder contains matching image and `.txt` label files.
- Class names are not empty.
- The class name order matches your label IDs.
- Input size, epochs, and batch size are positive integers.

### Model Weights Are Not Saved

If training completes successfully, check:

```text
<selected_output_folder>/<project_name>/weights/
```

If `best.pt` or `last.pt` is missing, read the training log for an earlier error.

### No Camera Is Listed

Try these steps:

- Close other apps that may be using the camera.
- Reconnect the camera.
- Press `Refresh`.
- Check OS camera privacy settings.
- Restart the app after connecting the camera.

## Project Structure

```text
main.py                 GUI entry point
src/gui_app.py          CustomTkinter GUI
src/gui_text.py         Japanese/English UI text
src/train.py            Training and output copy logic
src/detect.py           Image/video inference
src/camera.py           Camera inference
src/system_report.py    Training environment report
```

