import shutil
import os
import cv2
import mimetypes
from pathlib import Path
from ultralytics import YOLO
from typing import List, Union
from datetime import datetime

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

VALID_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
    '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV'
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

def is_valid_video(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid video by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_VIDEO_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('video/'):
            return False
            
        return True
    except Exception:
        return False

def normalize_path(file_path: Union[str, Path]) -> Path:
    """Normalize path to handle Japanese characters and different path formats"""
    try:
        return Path(file_path).resolve()
    except Exception:
        raise ValueError(f"Invalid path: {file_path}")

def get_media_files(directory: Union[str, Path]) -> tuple[List[Path], List[Path]]:
    """Recursively find all valid images and videos in a directory"""
    directory = normalize_path(directory)
    image_files = []
    video_files = []
    
    try:
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            if is_valid_image(file_path):
                image_files.append(file_path)
            elif is_valid_video(file_path):
                video_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return [], []
        
    return sorted(image_files), sorted(video_files)

def process_video(video_path: Path, model, output_dir: Path, conf_threshold: float = 0.5):
    """Process a video file and save detection results"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    video_name = video_path.stem
    video_output_dir = output_dir / f"{timestamp}_{video_name}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create video writer
    output_video_path = video_output_dir / f"{video_name}_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress (every 30 frames to avoid excessive printing)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing video: {progress:.1f}% ({frame_count}/{total_frames} frames)", end="")

        # Detect objects in frame
        results = model.predict(frame, save=False, conf=conf_threshold)
        
        # Only process frames with detections above threshold
        if len(results[0].boxes) > 0:
            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Save frame with detections
            frame_path = video_output_dir / f"frame_{detection_count:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)

            # Save detection results to txt
            txt_path = video_output_dir / f"frame_{detection_count:04d}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in results[0].boxes:
                    if box.conf[0] >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[int(box.cls[0])]
                        confidence = box.conf[0]
                        f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")
            
            detection_count += 1
            
            # Write frame to output video
            out.write(annotated_frame)
        else:
            # Write original frame to video if no detections
            out.write(frame)

        frame_count += 1

    # Clean up
    cap.release()
    out.release()
    
    print(f"\nVideo processing complete. {detection_count} frames with detections saved.")
    print(f"Output video saved to: {output_video_path}")
    
    return video_output_dir

def move_detection_results(source_dir, target_dir):
    source_dir = normalize_path(source_dir)
    target_dir = normalize_path(target_dir)
    
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
    
    # Find all valid images and videos in the folder
    images_folder = normalize_path(images_folder)
    image_files, video_files = get_media_files(images_folder)
    
    if not image_files and not video_files:
        print("No valid media files found in the directory")
        return

    results_dir = Path(images_folder) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    if image_files:
        # Convert Path objects to strings for YOLO predict
        image_paths = [str(path) for path in image_files]
        results = model.predict(image_paths, save=True, save_txt=True, imgsz=640, conf=0.5)
        
        runs_dir = Path('runs/detect')
        latest_run_dir = max(runs_dir.glob('*'), key=lambda p: p.stat().st_mtime)
        move_detection_results(latest_run_dir, results_dir)

    # Process videos
    for video_file in video_files:
        process_video(video_file, model, results_dir)

    if callback:
        callback(str(results_dir))