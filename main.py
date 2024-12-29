# version 0.1.1

import os
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, IntVar, Label, messagebox
from PIL import Image, ImageTk
import ctypes
import threading
import subprocess
import datetime
from queue import Queue, Empty
from src.train import create_yaml
from src.detect import detect_images
from src.camera import CameraDetection

project_name = ""
train_data_path = ""
model_save_path = ""
selected_model_size = ""
input_size = ""
epochs = ""
class_names = []
image_paths = []
current_image_index = 0
image_label = None
selected_model_var = None

global start_train_button, detection_progress_bar, image_index_label, camera_detection, detection_model_path, detection_save_dir, camera_id_entry

def get_screen_size():
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height

def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def read_output(process, queue):
    for line in iter(process.stdout.readline, b''):
        queue.put(line.decode('utf-8'))
    process.stdout.close()

def on_sidebar_select(window_title):
    clear_frame(main_frame)
    if window_title == "Train":
        show_ai_train_window()
    elif window_title == "Image/Video":
        show_image_detection_window()
    elif window_title == "Camera Detection":
        show_camera_detection_window()

output_queue = Queue()

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def update_output_textbox():
    try:
        line = output_queue.get_nowait()
        output_textbox.insert("end", line)
        output_textbox.yview_moveto(1)
    except Empty:
        pass
    finally:
        root.after(100, update_output_textbox)

def update_image():
    global current_image_index, image_label, image_paths, image_index_label
    if image_paths:
        image_index_text = f"{current_image_index + 1}/{len(image_paths)}"
        image_index_label.configure(text=image_index_text)
        img = Image.open(image_paths[current_image_index])
        img_w, img_h = img.size
        max_w, max_h = image_label.winfo_width(), image_label.winfo_height()  # Use the size of the image_label widget
        scale_w = max_w / img_w
        scale_h = max_h / img_h
        scale = min(scale_w, scale_h)
        img = img.resize((int(img_w * scale), int(img_h * scale)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo

def show_next_image():
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index + 1) % len(image_paths)
        update_image()

def show_prev_image():
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index - 1) % len(image_paths)
        update_image()

def start_training_and_capture_output(yaml_path, selected_model_size):
    global project_name, class_names, input_size, batch_size, epochs, model_save_path

    def run_training():
        nonlocal process
        if not all([project_name, train_data_path, class_names, model_save_path, selected_model_size, input_size, epochs, batch_size]):
            print("Error: One or more required parameters are missing.")
            return

        cmd_args = [
            'python', 'src/train.py',
            project_name, train_data_path, ','.join(class_names),
            model_save_path, selected_model_size, str(input_size),
            str(epochs), yaml_path, str(batch_size)
        ]

        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        threading.Thread(target=enqueue_output, args=(process.stdout, output_queue), daemon=True).start()
        process.wait()
        progress_bar.stop()

    process = None
    threading.Thread(target=run_training, daemon=True).start()
    progress_bar.start()

def show_ai_train_window():
    global project_name_entry, input_size_entry, epochs_entry, batch_size_entry, class_names_text, progress_bar, output_textbox, start_train_button, selected_model_var

    main_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

    # プロジェクト名入力
    ctk.CTkLabel(master=main_frame, text="Project Name: プロジェクト名（半角英数）", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.03, anchor=ctk.CENTER)
    project_name_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Project Name", width=250, height=50, font=("Roboto Medium", 18))
    project_name_entry.place(relx=0.2, rely=0.06, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # トレーニングデータ選択ボタン
    ctk.CTkLabel(master=main_frame, text="Select Train data: 学習データの選択", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.10, anchor=ctk.CENTER)
    train_data_button = ctk.CTkButton(master=main_frame, text="Select Train Data", command=select_train_data, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    train_data_button.place(relx=0.2, rely=0.13, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # モデル保存先選択ボタン
    ctk.CTkLabel(master=main_frame, text="Select Save Folder: モデルの保存先の選択", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.17, anchor=ctk.CENTER)
    model_save_button = ctk.CTkButton(master=main_frame, text="Select Model's Save Folder", command=select_model_save_folder, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    model_save_button.place(relx=0.2, rely=0.2, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # モデル選択ドロップダウン
    ctk.CTkLabel(master=main_frame, text="Select YOLO Model: YOLOのモデル選択", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.26, anchor=ctk.CENTER)
    model_options = ["YOLOv8-Nano", "YOLOv8-Small", "YOLOv8-Medium", "YOLOv8-Large", "YOLOv8-ExtraLarge", 
                     "YOLOv9-Compact", "YOLOv9-Enhanced",
                     "YOLOv10-Nano", "YOLOv10-Small", "YOLOv10-Medium", "YOLOv10-Balanced", "YOLOv10-Large", "YOLOv10-ExtraLarge",
                     "YOLOv11-Nano", "YOLOv11-Tiny", "YOLOv11-Medium","YOLOv11-Large","YOLOv11-ExtraLarge"]
    selected_model_var = ctk.StringVar(value=model_options[0])
    border_frame = ctk.CTkFrame(master=main_frame, fg_color="black", width=254, height=44)
    border_frame.place(relx=0.2, rely=0.29, anchor=ctk.CENTER)
    model_menu = ctk.CTkOptionMenu(
        master=border_frame,
        variable=selected_model_var,
        values=model_options,
        font=("Roboto Medium", 18),
        dropdown_font=("Roboto Medium", 18),
        button_color="white",
        button_hover_color="lightgray",
        dropdown_hover_color="lightgray",
        width=250,
        height=40,
    )
    model_menu.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

    # CNNの入力層のサイズ指定
    ctk.CTkLabel(master=main_frame, text="CNN Input Size: CNNの入力層のサイズ 【Ex: 640】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.39, anchor=ctk.CENTER)
    input_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Input Size", font=("Roboto Medium", 18))
    input_size_entry.place(relx=0.2, rely=0.42, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # エポック数
    ctk.CTkLabel(master=main_frame, text="Epochs: エポック数 【Ex: 100】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.46, anchor=ctk.CENTER)
    epochs_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Epochs", font=("Roboto Medium", 18))
    epochs_entry.place(relx=0.2, rely=0.49, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # バッチサイズ
    ctk.CTkLabel(master=main_frame, text="Batch Size: バッチサイズ 【Ex: 16】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.53, anchor=ctk.CENTER)
    batch_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Batch size", font=("Roboto Medium", 18))
    batch_size_entry.place(relx=0.2, rely=0.56, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # クラス名入力ウィンドウ
    ctk.CTkLabel(master=main_frame, text="Class name: クラス名", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.60, anchor=ctk.CENTER)
    class_names_text = ctk.CTkTextbox(master=main_frame, font=("Roboto Medium", 18))
    class_names_text.place(relx=0.2, rely=0.7, relwidth=0.3, relheight=0.17,  anchor=ctk.CENTER)

    # 学習開始ボタン
    start_train_button = ctk.CTkButton(master=main_frame, text="Start Training!", command=start_training, fg_color="chocolate1",border_color='black', border_width=3, font=("Roboto Medium", 44, "bold"), text_color='white')
    start_train_button.place(relx=0.2, rely=0.84, relwidth=0.4, relheight=0.08, anchor=ctk.CENTER)

    # トレーニング進捗表示ウィンドウ
    output_textbox = ctk.CTkTextbox(master=main_frame, corner_radius=20, font=("Roboto Medium", 14))
    output_textbox.place(relx=0.7, rely=0.45, relwidth=0.58, relheight=0.86, anchor=ctk.CENTER)

    # プログレスバー
    progress_bar = ctk.CTkProgressBar(master=main_frame, progress_color='limegreen', mode='indeterminate', indeterminate_speed=0.7)
    progress_bar.place(relx=0.5, rely=0.94, relwidth=0.7, anchor=ctk.CENTER)

def show_image_detection_window():
    global detection_images_folder_path, detection_model_path, image_label, detection_progress_bar, image_index_label
    clear_frame(main_frame)
    main_frame.pack(fill="both", expand=True)

    # 検出画像表示ウィンドウの設定
    image_label = Label(main_frame)
    image_label.place(relx=0.5, rely=0.44, relwidth=0.9, relheight=0.84, anchor=ctk.CENTER)

    # 画像フォルダの指定ボタンの設定
    select_images_folder_button = ctk.CTkButton(
        master=main_frame, 
        text="Select Image Folder", 
        command=select_detection_images_folder,
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 22),
        text_color='white',
    )
    select_images_folder_button.place(relx=0.05, rely=0.9, relwidth=0.15, relheight=0.05)

    # モデル選択ボタンの設定
    select_model_button = ctk.CTkButton(
        master=main_frame, 
        text="Select Model", 
        command=select_detection_model,
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 22),
        text_color='white',
    )
    select_model_button.place(relx=0.22, rely=0.9, relwidth=0.15, relheight=0.05)

    # 物体検出開始ボタンの設定
    start_detection_button = ctk.CTkButton(
        master=main_frame, 
        text="Start Detection!", 
        command=lambda: [detection_progress_bar.start(), start_image_detection()],
        fg_color="chocolate1",
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 34),
        text_color='white',
    )
    start_detection_button.place(relx=0.42, rely=0.89, relwidth=0.18, relheight=0.07)

    # 「前へ」ボタンの設定
    prev_button = ctk.CTkButton(master=main_frame, text="◀", command=show_prev_image, fg_color="DeepSkyBlue2", border_color='black', border_width=2, font=("Roboto Medium", 40), text_color='white')
    prev_button.place(relx=0.65, rely=0.9, relwidth=0.08, relheight=0.05)

    # 「次へ」ボタンの設定
    next_button = ctk.CTkButton(master=main_frame, text="▶", command=show_next_image, fg_color="DeepSkyBlue2", border_color='black', border_width=2, font=("Roboto Medium", 40), text_color='white')
    next_button.place(relx=0.75, rely=0.9, relwidth=0.08, relheight=0.05)

    # 画像インデックスを表示するラベルの初期化と配置
    image_index_label = ctk.CTkLabel(master=main_frame, text=" ", font=("Roboto Medium", 34))
    image_index_label.place(relx=0.85, rely=0.9, relwidth=0.1, relheight=0.05)

    # プログレスバーの設定
    detection_progress_bar = ctk.CTkProgressBar(master=main_frame, progress_color='limegreen', mode='indeterminate')
    detection_progress_bar.place(relx=0.5, rely=0.98, relwidth=0.7, anchor=ctk.CENTER)

def show_camera_detection_window():
    global camera_detection, detection_model_path, detection_save_dir, camera_id_entry, start_detection_button, image_label

    clear_frame(main_frame)
    main_frame.pack(fill="both", expand=True)

    camera_detection = None

    # Camera Stream Display
    image_label = Label(main_frame)
    image_label.place(relx=0.5, rely=0.48, relwidth=0.99, relheight=0.94, anchor=ctk.CENTER)

    # Select Model Button
    select_model_button = ctk.CTkButton(
        master=main_frame, 
        text="Select Model", 
        command=select_detection_model,
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 20),
        text_color='white',
    )
    select_model_button.place(relx=0.04, rely=0.96, relwidth=0.12, relheight=0.03)

    # Select Save Folder Button
    select_save_folder_button = ctk.CTkButton(
        master=main_frame, 
        text="Select Save Folder", 
        command=select_camera_save_folder,
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 20),
        text_color='white',
    )
    select_save_folder_button.place(relx=0.175, rely=0.96, relwidth=0.12, relheight=0.03)

    # Camera ID Entry
    camera_id_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Camera ID (Ex: 0)", font=("Roboto Medium", 18))
    camera_id_entry.place(relx=0.32, rely=0.96, relwidth=0.12, relheight=0.03)

    # Start Detection Button
    start_detection_button = ctk.CTkButton(
        master=main_frame, 
        text="START", 
        command=start_camera_detection,
        fg_color="green",
        border_color='black',
        border_width=2,
        font=("Roboto Medium", 28),
        text_color='white',
    )
    start_detection_button.place(relx=0.8, rely=0.96, relwidth=0.15, relheight=0.03)

    instructions_label = ctk.CTkLabel(
        master=main_frame, 
        text="Press ENTER to capture and save detection result.", 
        font=("Roboto Medium", 14)
    )
    instructions_label.place(relx=0.6, rely=0.98, anchor=ctk.CENTER)

    root.bind('<Return>', lambda event: save_callback())

    image_label.update_idletasks()
    image_label.update()

def select_train_data():
    global train_data_path
    train_data_path = filedialog.askdirectory()

def select_model_save_folder():
    global model_save_path
    model_save_path = filedialog.askdirectory()

def select_detection_images_folder():
    global detection_images_folder_path
    detection_images_folder_path = filedialog.askdirectory()
    if detection_images_folder_path:
        print(f"Selected folder: {detection_images_folder_path}")

def select_detection_model():
    global detection_model_path
    detection_model_path = filedialog.askopenfilename(filetypes=[("YOLOv8 Model", "*.pt")])
    if detection_model_path:
        print(f"Selected model: {detection_model_path}")

def select_camera_save_folder():
    global detection_save_dir
    detection_save_dir = filedialog.askdirectory()
    if detection_save_dir and camera_detection:
        camera_detection.set_save_directory(detection_save_dir)
        print(f"Selected save folder: {detection_save_dir}")

def select_detection_yaml():
    global detection_yaml_path
    detection_yaml_path = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
    if detection_yaml_path:
        print(f"Selected YAML: {detection_yaml_path}")

def animate_progress_bar(progress, step):
    if progress >= 100 or progress <= 0:
        step = -step

    progress_bar.set(progress)
    root.after(50, animate_progress_bar, progress + step, step)

def model_name_to_type(model_name):
    model_map = {
        "YOLOv8-Nano": "yolov8n", "YOLOv8-Small": "yolov8s", "YOLOv8-Medium": "yolov8m", "YOLOv8-Large": "yolov8l", "YOLOv8-ExtraLarge": "yolov8x",
        "YOLOv9-Compact": "yolov9c", "YOLOv9-Enhanced": "yolov9e",
        "YOLOv10-Nano": "yolov10n", "YOLOv10-Small": "yolov10s", "YOLOv10-Medium": "yolov10m", "YOLOv10-Balanced": "yolov10b", "YOLOv10-Large": "yolov10l", "YOLOv10-ExtraLarge": "yolov10x",
        "YOLOv11-Nano": "yolo11n", "YOLOv11-Small": "yolo11s", "YOLOv11-Medium": "yolo11m", "YOLOv11-Large": "yolo11l", "YOLOv11-ExtraLarge": "yolo11x",
    }
    return model_map.get(model_name, "")

def start_training():
    global project_name, train_data_path, model_save_path, selected_model_var, input_size, epochs, batch_size, class_names
    project_name = project_name_entry.get()
    input_size = input_size_entry.get()
    epochs = epochs_entry.get()
    batch_size = batch_size_entry.get()
    class_names = class_names_text.get("1.0", "end-1c").split('\n')
    class_names = [name for name in class_names if name.strip() != '']

    selected_model_size = model_name_to_type(selected_model_var.get())

    if not all([project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names]):
        print("Error: One or more required parameters are missing.")
        return

    yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)
    start_training_and_capture_output(yaml_path, selected_model_size)

def start_image_detection():
    global detection_images_folder_path, detection_model_path
    threading.Thread(target=detect_images, args=(detection_images_folder_path, detection_model_path, update_image_list), daemon=True).start()

def update_image_list(results_dir):
    global image_paths, current_image_index, detection_progress_bar
    image_paths = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.jpg') or f.endswith('.png')]
    current_image_index = 0
    update_image()
    detection_progress_bar.stop()

def start_camera_detection():
    global camera_detection, camera_id_entry, start_detection_button, image_label

    start_detection_button.configure(text="STOP", fg_color="red", command=stop_camera_detection)
    root.update()

    camera_id = int(camera_id_entry.get())
    try:
        camera_detection = CameraDetection(detection_model_path)
        camera_detection.start_camera(camera_id)
        camera_detection.show_camera_stream(image_label)
    except ValueError as e:
        image_label.config(text="No Camera", fg_color="red")
        start_detection_button.configure(text="START", fg_color="green", command=start_camera_detection)

def stop_camera_detection():
    global camera_detection, start_detection_button
    camera_detection.stop()
    start_detection_button.configure(text="START", fg_color="green", command=start_camera_detection)

def save_callback():
    if camera_detection:
        if detection_save_dir:
            camera_detection.set_save_directory(detection_save_dir)
        camera_detection.capture_frame()
    else:
        print("Camera detection not started")

def capture_frame(self):
    if not self.cap:
        return

    ret, frame = self.cap.read()
    if not ret:
        return

    self.scene_id += 1
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_filename = f"{timestamp}_{self.scene_id:04d}"

    origin_image_path = os.path.join(self.save_dir, f"{base_filename}_origin.jpg")
    cv2.imwrite(origin_image_path, frame)

    results = self.model(frame)
    self._draw_bounding_boxes(frame, results)

    detection_image_path = os.path.join(self.save_dir, f"{base_filename}_detection.jpg")
    cv2.imwrite(detection_image_path, frame)

    txt_path = os.path.join(self.save_dir, f"{base_filename}_detection.txt")
    with open(txt_path, 'w') as f:
        for result in results[0].boxes:
            if result.conf[0] >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                label = self.model.names[int(result.cls[0])]
                confidence = result.conf[0]
                f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

    return origin_image_path, detection_image_path, txt_path

def change_appearance_mode(new_appearance_mode):
    ctk.set_appearance_mode(new_appearance_mode)

screen_width, screen_height = get_screen_size()
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title('YOLO Train and Detect App')
root.geometry(f"{screen_width}x{screen_height}")

model_size_var = IntVar(value=1)

sidebar = ctk.CTkFrame(master=root, width=380, corner_radius=0)
sidebar.pack(side="left", fill="y")

main_frame = ctk.CTkFrame(master=root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

ai_creation_button = ctk.CTkButton(master=sidebar, text="Train", command=lambda: on_sidebar_select("Train"), fg_color="dodgerblue", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
ai_creation_button.pack(pady=10)

object_detection_button = ctk.CTkButton(master=sidebar, text="Image/Video", command=lambda: on_sidebar_select("Image/Video"), fg_color="chocolate1", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 20))
object_detection_button.pack(pady=10)

camera_detection_button = ctk.CTkButton(master=sidebar, text="Camera", command=lambda: on_sidebar_select("Camera Detection"), fg_color="chocolate1", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 20))
camera_detection_button.pack(pady=10)

app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv9", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)
app_name_label = ctk.CTkLabel(master=sidebar, text="&", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)
app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv8", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)

empty_space = ctk.CTkLabel(master=sidebar, text="")
empty_space.pack(fill=tk.BOTH, expand=True)

appearance_mode_var = ctk.StringVar(value="Light")
appearance_mode_label = ctk.CTkLabel(master=sidebar, text="Appearance Mode", font=("Roboto Medium", 12))
appearance_mode_label.pack(padx=10, pady=(0, 5), anchor='w')

light_mode_radio = ctk.CTkRadioButton(master=sidebar, text="Light", variable=appearance_mode_var, value="Light", command=lambda: change_appearance_mode("Light"))
light_mode_radio.pack(padx=10, pady=(0, 5), anchor='w')

dark_mode_radio = ctk.CTkRadioButton(master=sidebar, text="Dark", variable=appearance_mode_var, value="Dark", command=lambda: change_appearance_mode("Dark"))
dark_mode_radio.pack(padx=10, pady=(0, 10), anchor='w')

signature_label = ctk.CTkLabel(master=sidebar, text="© SpreadKnowledge 2024", text_color="white", font=("Roboto Medium", 10))
signature_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, anchor='w')

if __name__ == "__main__":
    root.after(100, update_output_textbox)
    root.mainloop()