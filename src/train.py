import os
import sys
import torch
import shutil
import glob
import random
import mimetypes
from pathlib import Path
from ultralytics import YOLO

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

def is_valid_image(file_path):
    """Check if a file is a valid image by examining extension and mime type"""
    try:
        file_path = Path(file_path)
        
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_IMAGE_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type is not None and mime_type.startswith('image/')
    except Exception:
        return False

def normalize_path(path):
    if not path:
        return path
    return str(Path(path).resolve())

def prepare_data(train_data_path):
    train_data_path = normalize_path(train_data_path)
    train_path = Path(train_data_path)
    train_dir_exists = (train_path / 'train/images').exists() and (train_path / 'train/labels').exists()
    val_dir_exists = (train_path / 'val/images').exists() and (train_path / 'val/labels').exists()

    if train_dir_exists and val_dir_exists:
        print("Train and validation directories already exist. Skipping file preparation.")
        return

    for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        (train_path / path).mkdir(parents=True, exist_ok=True)

    # Find all valid image files and their corresponding txt files
    paired_files = []
    for file_path in Path(train_data_path).iterdir():
        if file_path.is_file() and is_valid_image(str(file_path)):
            txt_file = file_path.with_suffix('.txt')
            if txt_file.exists():
                paired_files.append((file_path.name, txt_file.name))

    random.seed(0)
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * 0.8)
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    move_files(train_files, train_data_path, 'train')
    move_files(val_files, train_data_path, 'val')

def move_files(files, base_path, data_type):
    base_path = Path(base_path)
    for img_file, txt_file in files:
        # Move image file
        src_img = base_path / img_file
        dst_img = base_path / data_type / 'images' / img_file
        shutil.move(str(src_img), str(dst_img))

        # Move label file
        src_txt = base_path / txt_file
        dst_txt = base_path / data_type / 'labels' / txt_file
        shutil.move(str(src_txt), str(dst_txt))

def create_symlinks(files, base_path, data_type):
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        os.symlink(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        os.symlink(src_txt_path, dst_txt_path)

def clean_up(train_data_path):
    for path in ['train', 'val']:
        shutil.rmtree(os.path.join(train_data_path, path), ignore_errors=True)

def copy_and_remove_latest_run_files(model_save_path, project_name):
    model_save_path = Path(model_save_path)
    runs_path = Path('runs/detect') / project_name
    list_of_dirs = list(Path('runs/detect').glob(project_name))
    
    if not list_of_dirs:
        print(f"No 'runs/detect/{project_name}' directories found. Skipping copy and removal.")
        return

    latest_dir = max(list_of_dirs, key=lambda p: p.stat().st_mtime)

    if latest_dir.exists():
        for item in latest_dir.iterdir():
            dest = model_save_path / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest))

    runs_dir = Path('runs')
    if runs_dir.exists() and runs_dir.is_dir():
        shutil.rmtree(str(runs_dir))

def create_yaml(project_name, train_data_path, class_names, save_directory):
    prepare_data(train_data_path)

    train_path = str(Path(train_data_path) / 'train')
    val_path = str(Path(train_data_path) / 'val')

    # Ensure proper path format for YAML
    train_path = train_path.replace('\\', '/')
    val_path = val_path.replace('\\', '/')

    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: [{', '.join(f"'{name}'" for name in class_names)}]
"""
    print(f"Project Name: {project_name}")
    yaml_path = str(Path(save_directory) / f'{project_name}.yaml')
    print(f"YAML Path: {yaml_path}")
    
    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(yaml_content)
    return yaml_path

def train_yolo(data_yaml, model_type, img_size, batch, epochs, model_save_path, project_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(f'{model_type}.pt').to(device)
    results = model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=img_size, name=project_name, save=True)
    copy_and_remove_latest_run_files(model_save_path, project_name)
    clean_up(os.path.dirname(data_yaml))
    return results

def parse_args():
    project_name = sys.argv[1]
    train_data_path = sys.argv[2]
    class_names = sys.argv[3].split(',')
    model_save_path = sys.argv[4]
    model_type = sys.argv[5]
    img_size = int(sys.argv[6])
    epochs = int(sys.argv[7])
    yaml_path = sys.argv[8]
    batch_size = int(sys.argv[9])

    results = train_yolo(yaml_path, model_type, img_size, batch_size, epochs, model_save_path, project_name)
    print(f"Training completed. Model saved to {model_save_path}")

if __name__ == '__main__':
    parse_args()