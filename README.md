# YOLOv8_train_detection_GUI
You can easily run YOLOv8 train and detection with the GUI.

![GHwcFN9bUAAqcDW](https://github.com/SpreadKnowledge/YOLOv8_train_detection_GUI/assets/56751392/04b65806-57f2-4761-a8c4-9aee0623d550)
![GHwcF1rbwAAgG4y](https://github.com/SpreadKnowledge/YOLOv8_train_detection_GUI/assets/56751392/2fd83591-ab3c-467c-8606-d3f5a8d81c28)


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
