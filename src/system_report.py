import importlib
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))


LIBRARIES = [
    "python",
    "torch",
    "torchvision",
    "ultralytics",
    "customtkinter",
    "PIL",
    "cv2",
    "numpy",
]


def _module_version(module_name):
    if module_name == "python":
        return sys.version.replace("\n", " ")

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"not importable ({exc.__class__.__name__})"

    if module_name == "PIL":
        return getattr(module, "__version__", "unknown")
    return getattr(module, "__version__", "unknown")


def _run_command(args):
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except FileNotFoundError:
        return "not found"
    except Exception as exc:
        return f"failed: {exc}"

    output = (completed.stdout or completed.stderr or "").strip()
    if not output:
        return f"exit code {completed.returncode}, no output"
    return output


def collect_environment_report():
    lines = []
    lines.append("YOLO training environment report")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("[System]")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"Processor: {platform.processor() or 'unknown'}")
    lines.append(f"Python executable: {sys.executable}")
    lines.append("")

    lines.append("[Python libraries]")
    for library in LIBRARIES:
        lines.append(f"{library}: {_module_version(library)}")
    lines.append("")

    lines.append("[PyTorch CUDA]")
    try:
        import torch

        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        lines.append(f"torch.version.cuda: {torch.version.cuda}")
        lines.append(f"torch.backends.cudnn.version: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_gb = props.total_memory / (1024 ** 3)
                lines.append(
                    f"GPU {index}: {props.name}, compute capability "
                    f"{props.major}.{props.minor}, VRAM {total_gb:.2f} GB"
                )
    except Exception as exc:
        lines.append(f"PyTorch CUDA inspection failed: {exc}")
    lines.append("")

    lines.append("[NVIDIA driver]")
    lines.append(
        _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    )
    lines.append("")

    lines.append("[nvidia-smi]")
    lines.append(_run_command(["nvidia-smi"]))
    lines.append("")

    return "\n".join(lines)


def write_environment_report(output_dir, filename="training_environment.txt"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / filename
    report_path.write_text(collect_environment_report(), encoding="utf-8")
    return report_path
