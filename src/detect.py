import shutil
import os
from pathlib import Path
from ultralytics import YOLO

def move_detection_results(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_dir.iterdir():
        target_file = target_dir / file_path.name
        
        # Remove existing file/directory if it exists
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(str(target_file))
            else:
                target_file.unlink()
        
        # Move the file
        if file_path.is_file():
            shutil.move(str(file_path), str(target_file))
        else:
            shutil.move(str(file_path), str(target_dir))
    
    # Clean up source directory
    shutil.rmtree(str(source_dir))

def detect_images(images_folder, model_path, callback=None):
    model = YOLO(model_path)
    results = model.predict(images_folder, save=True, save_txt=True, imgsz=640, conf=0.5)
    
    runs_dir = Path('runs/detect')
    # Get latest run directory
    latest_run_dir = max(runs_dir.glob('*'), key=lambda p: p.stat().st_mtime)
    results_dir = Path(images_folder) / 'results'
    
    move_detection_results(latest_run_dir, results_dir)
    if callback:
        callback(str(results_dir))