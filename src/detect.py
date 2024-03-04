import shutil
import os
import glob
from ultralytics import YOLO

def move_detection_results(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        if os.path.exists(target_file):
            if os.path.isdir(target_file):
                shutil.rmtree(target_file)
            else:
                os.remove(target_file)
        shutil.move(source_file, target_dir)
    shutil.rmtree(source_dir)

def detect_images(images_folder, model_path, callback=None):
    model = YOLO(model_path)
    results = model.predict(images_folder, save=True, save_txt=True, imgsz=640, conf=0.5)
    latest_run_dir = max(glob.glob(os.path.join('runs', 'detect', '*')), key=os.path.getmtime)
    results_dir = os.path.join(images_folder, 'results')
    move_detection_results(latest_run_dir, results_dir)
    if callback:
        callback(results_dir)