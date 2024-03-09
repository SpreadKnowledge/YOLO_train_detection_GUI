import os
import sys
import torch
import shutil
import glob
import random
from ultralytics import YOLO

def prepare_data(train_data_path):
    train_dir_exists = os.path.exists(os.path.join(train_data_path, 'train/images')) and os.path.exists(os.path.join(train_data_path, 'train/labels'))
    val_dir_exists = os.path.exists(os.path.join(train_data_path, 'val/images')) and os.path.exists(os.path.join(train_data_path, 'val/labels'))

    if train_dir_exists and val_dir_exists:
        print("Train and validation directories already exist. Skipping file preparation.")
        return

    for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(os.path.join(train_data_path, path), exist_ok=True)

    all_files = set(os.listdir(train_data_path))
    paired_files = []
    for file in all_files:
        if file.endswith('.jpg') or file.endswith('.png'):
            basename = os.path.splitext(file)[0]
            txt_file = basename + '.txt'
            if txt_file in all_files:
                paired_files.append((file, txt_file))

    random.seed(0)
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * 0.8)
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    move_files(train_files, train_data_path, 'train')
    move_files(val_files, train_data_path, 'val')

def move_files(files, base_path, data_type):
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        shutil.move(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        shutil.move(src_txt_path, dst_txt_path)

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
    list_of_dirs = glob.glob('runs/detect/' + project_name)
    if not list_of_dirs:
        print("No 'runs/detect/" + project_name + "' directories found. Skipping copy and removal.")
        return

    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    if os.path.exists(latest_dir):
        for item in os.listdir(latest_dir):
            s = os.path.join(latest_dir, item)
            d = os.path.join(model_save_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    runs_dir = 'runs'
    if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
        shutil.rmtree(runs_dir)

def create_yaml(project_name, train_data_path, class_names, save_directory):
    prepare_data(train_data_path)

    train_path = os.path.join(train_data_path, 'train').replace('\\', '/')
    val_path = os.path.join(train_data_path, 'val').replace('\\', '/')

    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: [{', '.join(f"'{name}'" for name in class_names)}]
"""
    print(f"Project Name: {project_name}")
    yaml_path = os.path.join(save_directory, f'{project_name}.yaml').replace('\\', '/')
    print(f"YAML Path: {yaml_path}")
    with open(yaml_path, 'w') as file:
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