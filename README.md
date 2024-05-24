# YOLO_train_detection_GUI
You can easily run YOLOv9 and YOLOv8 train and detection with the GUI.

↓ YouTube has a comprehensive guide on how to use this GUI.

[![Hou to use this App](https://img.youtube.com/vi/Jk-JkBn4Na0/maxresdefault.jpg)](https://youtu.be/Jk-JkBn4Na0?si=iZsowHXhxP2pG_XZ)

# Prepare
Since we will be using PyTorch 2.2.1, please install CUDA 12.1 or CUDA 11.8 beforehand. Note that main.py will work without CUDA.

## Download repository and create virtual environment
```
conda create -n yolo-gui python=3.12
conda activate yolo-gui

git clone https://github.com/SpreadKnowledge/YOLO_train_detection_GUI.git
cd YOLO_train_detection_GUI
pip install -r requirements.txt
```
# Prepare your original train dataset
Please save the YOLO format's annotation file (.txt) created by labelImg or VoTT together with the paired image files in one directory.
You do not need to split them into train and val directories or create yaml files. The process is handled automatically within this application.

# Run GUI application 
```
python main.py
```
