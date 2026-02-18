import cv2
import os
import numpy as np


def draw_boxes_on_image(img_path, label_path, output_img_path):
    """
    读取图片和YOLO标签，在图上画框并保存
    """
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  无法读取图片: {img_path}")
        return

    h, w = img.shape[:2]

    # 2. 读取标签（如果没有标签就直接保存原图）
    if not os.path.exists(label_path):
        cv2.imwrite(output_img_path, img)
        return

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 3. 逐行画框
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        # YOLO 转像素
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 画框（绿色，厚度2）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 可选：在框左上角写类别ID
        cv2.putText(img, str(cls_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4. 保存画好框的图
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, img)
    print(f"✅ 已画框并保存: {output_img_path}")


def batch_draw_boxes(input_root, output_root):
    """
    批量对整个数据集画框
    input_root: 输入根目录（下有 images/ 和 labels/）
    output_root: 输出根目录（下会生成 images/ 画框图）
    """
    input_images = os.path.join(input_root, "images")
    input_labels = os.path.join(input_root, "labels")

    output_images = os.path.join(output_root, "images")
    os.makedirs(output_images, exist_ok=True)

    # 遍历所有图片
    for img_name in os.listdir(input_images):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(input_images, img_name)
        base, _ = os.path.splitext(img_name)
        label_path = os.path.join(input_labels, base + ".txt")
        out_path = os.path.join(output_images, img_name)

        draw_boxes_on_image(img_path, label_path, out_path)

    print("\n📊 批量画框完成！")
    print(f"📂 画框后图片保存在: {output_images}")


if __name__ == '__main__':
    # ====================== 你只需要改这两个路径 ======================
    INPUT_DATASET_ROOT = r"E:\workspace-pycharm\ttq_dataset\data\test_after"
    OUTPUT_VIS_ROOT = r"E:\workspace-pycharm\ttq_dataset\data\val"
    # ================================================================

    batch_draw_boxes(INPUT_DATASET_ROOT, OUTPUT_VIS_ROOT)