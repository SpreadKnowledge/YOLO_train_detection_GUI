# YOLOv8_train_detection_GUI
You can easily run YOLOv8 train and detection with the GUI.
# Prepare
Since we will be using PyTorch 2.2.1, please install CUDA 12.1 or CUDA 11.8 beforehand. Note that main.py will work without CUDA.

## Download repository and create virtual environment
```
conda create -n yolov8 python=3.11
conda activate yolov8

git clone https://github.com/SpreadKnowledge/YOLOv8_train_detection_GUI.git
cd YOLOv8_train_detection_GUI
pip install -r requirements.txt
```
# Prepare your original train dataset
Please save the YOLO format's annotation file (.txt) created by labelImg or VoTT together with the paired image files in one directory.
You do not need to split them into train and val directories or create yaml files. The process is handled automatically within the application.

# Run YOLOv8 GUI application 
```
python main.py
```
