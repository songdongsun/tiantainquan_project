import cv2
import numpy as np
import os

def process_one_image(img_path, save_path):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return False

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化（黑底 → 0，主体 → 255）
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # 找轮廓（只找最外层）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # 取面积最大的轮廓（主体）
    cnt = max(contours, key=cv2.contourArea)

    # 外接矩形
    x, y, w, h = cv2.boundingRect(cnt)

    # 裁切
    cropped = img[y:y+h, x:x+w]

    # 保存
    cv2.imwrite(save_path, cropped)
    return True

def batch_crop(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(exts):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            ok = process_one_image(in_path, out_path)
            print(f"{'OK' if ok else 'FAIL'}: {fname}")

# ====================== 你只需要改这里 ======================
INPUT_DIR = r"E:\workspace-pycharm\ttq_dataset\data\PT_imgs"    # 你的原图文件夹
OUTPUT_DIR = r"data/dataset3/images"  # 裁切后保存的文件夹
# ============================================================

if __name__ == "__main__":
    batch_crop(INPUT_DIR, OUTPUT_DIR)