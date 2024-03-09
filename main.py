# version 0.1.0

import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, IntVar, Label
from PIL import Image, ImageTk
import ctypes
import threading
import subprocess
from queue import Queue, Empty
from src.train import create_yaml
from src.detect import detect_images

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

global start_train_button, detection_progress_bar, image_index_label

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
    elif window_title == "Detect":
        show_image_detection_window()

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

def start_training_and_capture_output(yaml_path):
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
    global project_name_entry, input_size_entry, epochs_entry, batch_size_entry, class_names_text, progress_bar, model_size_var, output_textbox, start_train_button
    # AI作成ウィンドウのGUI要素を配置
    main_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

    # プロジェクト名入力
    ctk.CTkLabel(master=main_frame, text="プロジェクト名（半角英数）", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.03, anchor=ctk.CENTER)
    project_name_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Project Name", width=250, height=50, font=("Roboto Medium", 18))
    project_name_entry.place(relx=0.2, rely=0.06, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # トレーニングデータ選択ボタン
    ctk.CTkLabel(master=main_frame, text="学習データの選択", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.10, anchor=ctk.CENTER)
    train_data_button = ctk.CTkButton(master=main_frame, text="Select Train Data", command=select_train_data, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    train_data_button.place(relx=0.2, rely=0.13, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # モデル保存先選択ボタン
    ctk.CTkLabel(master=main_frame, text="モデルの保存先の選択", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.17, anchor=ctk.CENTER)
    model_save_button = ctk.CTkButton(master=main_frame, text="Select Model's Save Folder", command=select_save_folder, border_color='black', border_width=2, font=("Roboto Medium", 24), text_color='white')
    model_save_button.place(relx=0.2, rely=0.2, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # YOLOv9のモデルサイズ
    ctk.CTkLabel(master=main_frame, text="YOLOv9のモデルサイズの選択", font=("Roboto Medium", 18)).place(relx=0.1, rely=0.26, anchor=tk.W)
    initial_relx_v9 = 0.04
    step_size_v9 = 0.07
    model_sizes_v9 = [("Compact", "c", 1), ("Enhanced", "e", 2)]
    for index, (text, model_code, value) in enumerate(model_sizes_v9, start=1):
        ctk.CTkRadioButton(master=main_frame, text=text, variable=model_size_var, value=value, fg_color='deep sky blue').place(relx=initial_relx_v9 + step_size_v9 * (index - 1), rely=0.28)

    # YOLOv8のモデルサイズ
    ctk.CTkLabel(master=main_frame, text="YOLOv8のモデルサイズの選択", font=("Roboto Medium", 18)).place(relx=0.1, rely=0.32, anchor=tk.W)
    initial_relx_v8 = 0.04
    step_size_v8 = 0.07
    model_sizes_v8 = [("Nano", "n", 3), ("Small", "s", 4), ("Medium", "m", 5), ("Large", "l", 6), ("ExtraLarge", "x", 7)]
    for index, (text, model_code, value) in enumerate(model_sizes_v8, start=1):
        ctk.CTkRadioButton(master=main_frame, text=text, variable=model_size_var, value=value, fg_color='deep sky blue').place(relx=initial_relx_v8 + step_size_v8 * (index - 1), rely=0.34)

    # CNNの入力層のサイズ指定
    ctk.CTkLabel(master=main_frame, text="CNNの入力層のサイズ 【例：640】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.39, anchor=ctk.CENTER)
    input_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Input Size", font=("Roboto Medium", 18))
    input_size_entry.place(relx=0.2, rely=0.42, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # エポック数
    ctk.CTkLabel(master=main_frame, text="エポック数 【例：100】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.46, anchor=ctk.CENTER)
    epochs_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Epochs", font=("Roboto Medium", 18))
    epochs_entry.place(relx=0.2, rely=0.49, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # バッチサイズ
    ctk.CTkLabel(master=main_frame, text="バッチサイズ 【例：16】", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.53, anchor=ctk.CENTER)
    batch_size_entry = ctk.CTkEntry(master=main_frame, placeholder_text="Batch size", font=("Roboto Medium", 18))
    batch_size_entry.place(relx=0.2, rely=0.56, relwidth=0.3, relheight=0.04, anchor=ctk.CENTER)

    # クラス名入力ウィンドウ
    ctk.CTkLabel(master=main_frame, text="クラス名の入力", font=("Roboto Medium", 18)).place(relx=0.2, rely=0.60, anchor=ctk.CENTER)
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

def select_train_data():
    global train_data_path
    train_data_path = filedialog.askdirectory()

def select_save_folder():
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

def start_training():
    global project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names
    project_name = project_name_entry.get()
    input_size = input_size_entry.get()
    epochs = epochs_entry.get()
    batch_size = batch_size_entry.get()
    class_names = class_names_text.get("1.0", "end-1c").split('\n')
    class_names = [name for name in class_names if name.strip() != '']

    model_size_options = {1: "yolov9c", 2: "yolov9e", 3: "yolov8n", 4: "yolov8s", 5: "yolov8m", 6: "yolov8l", 7: "yolov8x"}
    selected_model_size = model_size_options[model_size_var.get()]

    if not all([project_name, train_data_path, model_save_path, selected_model_size, input_size, epochs, batch_size, class_names]):
        print("Error: One or more required parameters are missing.")
        return

    yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)
    start_training_and_capture_output(yaml_path)

def start_image_detection():
    global detection_images_folder_path, detection_model_path
    threading.Thread(target=detect_images, args=(detection_images_folder_path, detection_model_path, update_image_list), daemon=True).start()

def update_image_list(results_dir):
    global image_paths, current_image_index, detection_progress_bar
    image_paths = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.jpg') or f.endswith('.png')]
    current_image_index = 0
    update_image()
    detection_progress_bar.stop()

screen_width, screen_height = get_screen_size()
ctk.set_appearance_mode("Dark")
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

object_detection_button = ctk.CTkButton(master=sidebar, text="Detect", command=lambda: on_sidebar_select("Detect"), fg_color="dodgerblue", text_color="white", border_color='black', border_width=2, font=("Roboto Medium", 24))
object_detection_button.pack(pady=10)

app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv9", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)
app_name_label = ctk.CTkLabel(master=sidebar, text="&", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)
app_name_label = ctk.CTkLabel(master=sidebar, text="YOLOv8", font=("Roboto Medium", 16))
app_name_label.pack(pady=1)

signature_label = ctk.CTkLabel(master=sidebar, text="© SpreadKnowledge 2024", text_color="white", font=("Roboto Medium", 10))
signature_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5, anchor='w')

if __name__ == "__main__":
    root.after(100, update_output_textbox)
    root.mainloop()