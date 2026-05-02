import os
import sys
import shutil
import random
import mimetypes
import time
import json
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.system_report import write_environment_report

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
        return False

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
    (train_path / ".yolo_gui_prepared").write_text("created by YOLO GUI\n", encoding="utf-8")
    return True

def move_files(files, base_path, data_type):
    base_path = Path(base_path)
    for img_file, txt_file in files:
        src_img = base_path / img_file
        dst_img = base_path / data_type / 'images' / img_file
        shutil.copy2(str(src_img), str(dst_img))

        src_txt = base_path / txt_file
        dst_txt = base_path / data_type / 'labels' / txt_file
        shutil.copy2(str(src_txt), str(dst_txt))

def create_symlinks(files, base_path, data_type):
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        os.symlink(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        os.symlink(src_txt_path, dst_txt_path)

def clean_up(train_data_path):
    marker = Path(train_data_path) / ".yolo_gui_prepared"
    if not marker.exists():
        return
    for path in ['train', 'val']:
        shutil.rmtree(os.path.join(train_data_path, path), ignore_errors=True)
    marker.unlink(missing_ok=True)

def _destination_dir(model_save_path, project_name):
    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    if model_save_path.name == project_name:
        return model_save_path
    return model_save_path / project_name


def copy_training_run_files(run_dir, model_save_path, project_name):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run directory was not found: {run_dir}")

    destination = _destination_dir(model_save_path, project_name)
    destination.mkdir(parents=True, exist_ok=True)

    if run_dir.resolve() == destination.resolve():
        return destination

    for item in run_dir.iterdir():
        dest = destination / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(dest))

    return destination

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
names: {json.dumps(class_names, ensure_ascii=False)}
"""
    print(f"Project Name: {project_name}")
    yaml_path = str(Path(save_directory) / f'{project_name}.yaml')
    print(f"YAML Path: {yaml_path}")
    
    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(yaml_content)
    return yaml_path

def train_yolo(data_yaml, model_type, img_size, batch, epochs, model_save_path, project_name, train_data_path=None):
    import torch
    import ultralytics
    from ultralytics import YOLO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GUI_DEVICE device={device} torch_cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}", flush=True)
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Using CPU: CUDA GPU was not detected by PyTorch.", flush=True)

    model_file = f'{model_type}.pt'
    try:
        model = YOLO(model_file).to(device)
    except FileNotFoundError as exc:
        print(
            f"Model file '{model_file}' was not found, and Ultralytics "
            f"{ultralytics.__version__} did not auto-download it."
        )
        print("If this is a YOLO26 model, upgrade Ultralytics and try again:")
        print(f"{sys.executable} -m pip install -U ultralytics")
        raise exc

    start_time = time.time()

    def report_epoch_progress(trainer):
        completed = int(getattr(trainer, "epoch", 0)) + 1
        total = int(getattr(trainer, "epochs", epochs))
        elapsed = max(time.time() - start_time, 0.1)
        eta = max(total - completed, 0) * (elapsed / max(completed, 1))
        print(
            f"GUI_PROGRESS epoch={completed} total={total} "
            f"elapsed={elapsed:.1f} eta={eta:.1f}",
            flush=True,
        )

    model.add_callback("on_fit_epoch_end", report_epoch_progress)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=str(Path("runs") / "detect"),
        name=project_name,
        save=True,
        device=device,
    )

    run_dir = Path(getattr(model.trainer, "save_dir", ""))
    destination = copy_training_run_files(run_dir, model_save_path, project_name)
    data_yaml_path = Path(data_yaml)
    yaml_destination = destination / data_yaml_path.name
    if data_yaml_path.exists() and data_yaml_path.resolve() != yaml_destination.resolve():
        shutil.copy2(str(data_yaml_path), str(yaml_destination))
    report_path = write_environment_report(destination)

    weights_dir = destination / "weights"
    best_weight = weights_dir / "best.pt"
    last_weight = weights_dir / "last.pt"
    if best_weight.exists() or last_weight.exists():
        print(f"GUI_ARTIFACT weights={best_weight if best_weight.exists() else last_weight}", flush=True)
    else:
        print(f"WARNING: Weight files were not found in {weights_dir}", flush=True)
    print(f"GUI_ARTIFACT environment={report_path}", flush=True)
    print(f"Training output copied to: {destination}", flush=True)

    if train_data_path:
        clean_up(train_data_path)
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

    results = train_yolo(
        yaml_path,
        model_type,
        img_size,
        batch_size,
        epochs,
        model_save_path,
        project_name,
        train_data_path,
    )
    print(f"Training completed. Model saved to {model_save_path}")

if __name__ == '__main__':
    parse_args()
