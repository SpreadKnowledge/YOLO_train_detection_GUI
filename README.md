
# YOLO Train and Detect App “Complete Your Learning and Detection Seamlessly on This GUI”

🎉 **NEW: 2024/12/29 ** We now support YOLOv11! Train and detect with the latest YOLO version. 🎉

![スクリーンショット 2024-05-25 163018](https://github.com/SpreadKnowledge/YOLO_train_detection_GUI/assets/56751392/5ff31879-8756-4561-ad5e-a5f6b0529798)

This application is a user-friendly GUI tool built with PyTorch, Ultralytics library, and CustomTkinter. It allows you to easily develop and train YOLOv8 and YOLOv9 models, and perform object detection on images, videos, and webcam feeds using the trained models. The detection results can be saved for further analysis.

↓ Please watch the instructional video (in English) uploaded on YouTube to check out the specific operation.
[![YOLO Train and Detect App Demo](https://img.youtube.com/vi/Jk-JkBn4Na0/0.jpg)](https://youtu.be/Jk-JkBn4Na0?si=hMqGkJ4YAjnaKbQW)

## Environment Setup

### Using venv

1. Clone this repository:
```bash
git clone https://github.com/SpreadKnowledge/YOLO_train_detection_GUI.git
```
2. Navigate to the project directory:
```bash
cd your-repository
```
3. Create a virtual environment:
```bash
python -m venv venv
```
4. Activate the virtual environment:
- For Windows:
  ```
  venv\Scripts\activate
  ```
- For macOS and Linux:
  ```
  source venv/bin/activate
  ```

5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Using Anaconda

1. Clone this repository:
```bash
git clone https://github.com/SpreadKnowledge/YOLO_train_detection_GUI.git
```
2. Navigate to the project directory:
```bash
cd your-repository
```
3. Create a new Anaconda environment:
```bash
conda create --name yolo-app python=3.12
```
4. Activate the Anaconda environment:
```bash
conda activate yolo-app
```
5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Preparing Training Data

Before training your YOLO model, you need to prepare the training data in the YOLO format. For each image, you should have a corresponding text file with the same name containing the object annotations. The text file should follow the format:
```plaintext
<class_id> <x_center> <y_center> <width> <height>
```
- `<class_id>`: Integer representing the class ID of the object.
- `<x_center>`, `<y_center>`: Floating-point values representing the center coordinates of the object bounding box, normalized by the image width and height.
- `<width>`, `<height>`: Floating-point values representing the width and height of the object bounding box, normalized by the image width and height.

Place the image files and their corresponding annotation text files in the same directory.

## Running the Application

To run the YOLO Train and Detect App, execute the following command:
```bash
python main.py
```

## Application Features

### Train Tab

In the Train tab, you can train your own YOLO model:

1. Enter a project name (alphanumeric only).
2. Select the directory containing your training data (images and annotation text files).
3. Choose the directory where you want to save the trained model.
4. Select the model size for YOLOv9 (Compact or Enhanced) or YOLOv8 (Nano, Small, Medium, Large, or ExtraLarge).
5. Specify the input size for the CNN (e.g., 640).
6. Set the number of epochs for training (e.g., 100).
7. Enter the batch size for training (e.g., 16).
8. Input the class names, one per line, in the provided text box.
9. Click the "Start Training!" button to begin the training process.

Note: Make sure to provide all the required information, or the training process will not start.

### Image/Video Tab

In the Image/Video tab, you can perform object detection on images or videos:

1. Select the folder containing the images or videos you want to process.
2. Choose the trained YOLO model file (.pt) for detection.
3. Click the "Start Detection!" button to initiate the detection process.
4. The detection results will be displayed in the application window.
5. Use the navigation buttons (◀ and ▶) to browse through the processed images or video frames.

Note: Ensure that you have selected the correct folders and model file, or the detection process will not work.

### Camera Tab

In the Camera tab, you can perform real-time object detection using a webcam:

1. Select the trained YOLO model file (.pt) for detection.
2. Choose the directory where you want to save the detection results.
3. Enter the camera ID (e.g., 0 for the default webcam).
4. Click the "START" button to begin the real-time detection.
5. The live camera feed with object detection will be displayed in the application window.
6. Press the "ENTER" key to capture and save the current frame and its detection results.
7. Click the "STOP" button to stop the real-time detection.

Note: Make sure that you have selected the correct model file and save directory, and entered a valid camera ID, or the real-time detection will not function properly.
