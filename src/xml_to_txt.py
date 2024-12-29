"""アノテーションデータをxml形式からtxt形式に変換する
"""

import os
import xml.etree.ElementTree as ET


xml_dir = r"C:\Users\he81t\ubuntu\yamaguchi\fukuoka_chicken\original_data\Annotations_pascal_xml"  # XMLファイルがあるディレクトリ
txt_dir = r"C:\Users\he81t\ubuntu\yamaguchi\fukuoka_chicken\original_data\Annotations_yolo_txt"    # YOLO形式のtxtファイルを出力するディレクトリ

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# クラス一覧を格納するリスト
classes = []

# xml_dir 内のファイルを走査
for filename in os.listdir(xml_dir):
    # 拡張子が .xml のファイルだけ処理
    if filename.endswith(".xml"):
        # XMLファイルのパス
        xml_path = os.path.join(xml_dir, filename)
        
        # XMLをパース
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 画像サイズを取得
        size = root.find("size")
        if size is None:
            # sizeタグがない場合はスキップ
            continue
        
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        
        # 出力テキストファイルのパス (.txtに置き換え)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)
        
        # YOLO形式のアノテーションを書き込むためのリスト
        yolo_annotations = []
        
        # objectタグをすべて取得し、YOLO形式に変換
        for obj in root.findall("object"):
            # クラス名
            class_name = obj.find("name").text
            if class_name not in classes:
                classes.append(class_name)
            class_id = classes.index(class_name)
            
            # バウンディングボックス座標
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            # YOLO形式は (class_id, x_center, y_center, w, h) [正規化済み]
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            # 小数点以下の桁数を整えて書き出す場合はここでformatを利用
            annotation_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(annotation_str)
        
        # テキストファイルに書き出し
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_annotations:
                f.write(line + "\n")

# 変換が終わった後、classesに格納されているクラス一覧を確認したい場合は、別途出力することもできます。
print("クラス一覧:", classes)