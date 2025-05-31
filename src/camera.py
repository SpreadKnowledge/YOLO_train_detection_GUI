import cv2
import threading
import time
import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime
from PIL import Image, ImageTk

def normalize_path(path):
    if not path:
        return path
    return str(Path(path).resolve())

class CameraDetection:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.cap = None
        self.running = False
        self.save_dir = ""
        self.scene_id = 0

    def start_camera(self, camera_id):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("Unable to open camera")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_save_directory(self, directory):
        self.save_dir = directory

    def show_camera_stream(self, display_label):
        self.running = True
        threading.Thread(target=self._update_stream, args=(display_label,), daemon=True).start()

    def _update_stream(self, display_label):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)
            detection_frame = frame.copy()
            self._draw_bounding_boxes(detection_frame, results)

            img = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            img = self._resize_image_to_fit(img, display_label.winfo_width(), display_label.winfo_height())
            img = Image.fromarray(img)

            img = ImageTk.PhotoImage(image=img)
            display_label.config(image=img)
            display_label.image = img

            display_label.update_idletasks()
            display_label.update()

            time.sleep(0.03)

    def _resize_image_to_fit(self, image, max_width, max_height):
        height, width = image.shape[:2]
        aspect_ratio = width / height

        if aspect_ratio > max_width / max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image

    def _draw_bounding_boxes(self, frame, results):
        colors = {}
        for result in results[0].boxes:
            if result.conf[0] >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = self.model.names[int(result.cls[0])]
                confidence = result.conf[0]

                if label not in colors:
                    colors[label] = (int(hash(label) % 255), int((hash(label) * 2) % 255), int((hash(label) * 3) % 255))

                color = colors[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def capture_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.scene_id += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{timestamp}_{self.scene_id:04d}"

        # Use pathlib for path handling
        save_dir_path = Path(self.save_dir)
        
        # オリジナル画像を保存（PNG形式で保存してクオリティを維持）
        origin_image_path = str(save_dir_path / f"{base_filename}_origin.png")
        cv2.imwrite(origin_image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # 検出結果を描画
        results = self.model(frame)
        self._draw_bounding_boxes(frame, results)

        # 検出結果画像を保存（JPEG形式で容量を抑える）
        detection_image_path = str(save_dir_path / f"{base_filename}_detection.jpg")
        cv2.imwrite(detection_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 検出結果のテキストを保存
        txt_path = str(save_dir_path / f"{base_filename}_detection.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for result in results[0].boxes:
                if result.conf[0] >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    label = self.model.names[int(result.cls[0])]
                    confidence = result.conf[0]
                    f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

        return origin_image_path, detection_image_path, txt_path

    def stop(self):
        self.running = False
        self.stop_camera()