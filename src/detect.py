import shutil
import os
import mimetypes
from pathlib import Path
from ultralytics import YOLO
from typing import List, Union

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

def is_valid_image(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid image by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_IMAGE_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('image/'):
            return False
            
        return True
    except Exception:
        return False

def normalize_image_path(file_path: Union[str, Path]) -> Path:
    """Normalize image path to handle Japanese characters and different path formats"""
    try:
        return Path(file_path).resolve()
    except Exception:
        raise ValueError(f"Invalid path: {file_path}")

def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """Recursively find all valid images in a directory"""
    directory = normalize_image_path(directory)
    image_files = []
    
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file() and is_valid_image(file_path):
                image_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return []
        
    return sorted(image_files)

def move_detection_results(source_dir, target_dir):
    source_dir = normalize_image_path(source_dir)
    target_dir = normalize_image_path(target_dir)
    
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
    
    # Find all valid images in the folder recursively using the utility function
    images_folder = normalize_image_path(images_folder)
    image_files = get_image_files(images_folder)
    
    if not image_files:
        print("No valid images found in the directory")
        return
        
    # Convert Path objects to strings for YOLO predict
    image_paths = [str(path) for path in image_files]
    
    # Process found images
    results = model.predict(image_paths, save=True, save_txt=True, imgsz=640, conf=0.5)
    
    runs_dir = Path('runs/detect')
    # Get latest run directory
    latest_run_dir = max(runs_dir.glob('*'), key=lambda p: p.stat().st_mtime)
    results_dir = Path(images_folder) / 'results'
    
    move_detection_results(latest_run_dir, results_dir)
    if callback:
        callback(str(results_dir))