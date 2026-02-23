import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from ultralytics.data.augment import BaseTransform, WaveAugment
from ultralytics.utils.instance import Instances  # YOLOv8标注格式类
from ultralytics.data.augment import WaveAugment
import os
import torch


# ===================== YOLO标签解析工具（保留） =====================
def yolo2xyxy(yolo_line: str, img_w: int, img_h: int) -> list:
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        return None

    try:
        cls_id = int(parts[0])
        x_c = float(parts[1]) * img_w
        y_c = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
    except ValueError:
        return None

    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return [x1, y1, x2, y2, cls_id]


def read_yolo_labels(label_path: str, img_w: int, img_h: int) -> tuple[list, list]:
    if not os.path.exists(label_path):
        print(f"警告：标签文件不存在 {label_path}")
        return [], []

    xyxy_list = []
    cls_list = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            result = yolo2xyxy(line, img_w, img_h)
            if result is None:
                continue
            x1, y1, x2, y2, cls_id = result
            xyxy_list.append([x1, y1, x2, y2])
            cls_list.append(cls_id)

    return xyxy_list, cls_list


# ===================== 测试代码（完全绕开BaseTransform的__call__返回值） =====================
if __name__ == "__main__":
    # ========== 1. 配置参数 ==========
    INPUT_IMAGE_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\pt_ng\images\train\1 (2161)_2.jpg"  # 你的图片路径
    LABEL_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\pt_ng\labels\train\1 (2161)_2.txt"  # YOLO标签路径
    OUTPUT_IMAGE_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\bd"  # 保存路径
    AMPLITUDE = 8.0
    FREQUENCY = 0.5
    DIRECTION = "horizontal"


    # ========== 2. 模拟Instances类 ==========
    class MockInstances:
        def __init__(self, xyxy, cls):
            self.xyxy = torch.tensor(xyxy, dtype=torch.float32) if xyxy else torch.empty((0, 4))
            self.cls = torch.tensor(cls, dtype=torch.float32) if cls else torch.empty((0,))

        def __len__(self):
            return len(self.xyxy)


    # ========== 3. 读取图片 ==========
    original_img = cv2.imread(INPUT_IMAGE_PATH)
    if original_img is None:
        raise ValueError(f"无法读取图片，请检查路径：{INPUT_IMAGE_PATH}")
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img_rgb.shape[:2]
    print(f"✅ 成功读取图片：尺寸 {w}x{h}")

    # ========== 4. 读取YOLO标签 ==========
    xyxy_list, cls_list = read_yolo_labels(LABEL_PATH, w, h)
    print(f"✅ 成功读取 {len(xyxy_list)} 个标注框")
    test_instances = MockInstances(xyxy=xyxy_list, cls=cls_list)

    # ========== 5. 手动应用波动增强 ==========
    wave_aug = WaveAugment(amplitude=AMPLITUDE, frequency=FREQUENCY, direction=DIRECTION)
    # 变换图像
    aug_img_rgb, map_x, map_y = wave_aug.warp_image(original_img_rgb)
    # 变换标注框
    aug_instances = wave_aug.warp_instances(test_instances, map_x, map_y)

    # ========== 6. 绘制标注框并保存图片 ==========
    # 增强后图片转BGR（适配cv2保存）
    aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)

    # 绘制增强后的标注框（红色，线宽2）
    aug_xyxy = aug_instances.xyxy.numpy()
    for box in aug_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(aug_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 保存增强后的图片（带标注框）
    cv2.imwrite(OUTPUT_IMAGE_PATH, aug_img_bgr)
    print(f"✅ 增强后的图片已保存至：{os.path.abspath(OUTPUT_IMAGE_PATH)}")

    # ========== 7. 额外保存原始图（带标注框，方便对比） ==========
    original_img_bgr = original_img.copy()
    for box in xyxy_list:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    original_save_path = "original_with_box.jpg"
    cv2.imwrite(original_save_path, original_img_bgr)
    print(f"✅ 原始图片（带标注框）已保存至：{os.path.abspath(original_save_path)}")

    # ========== 8. 输出坐标对比（关键信息） ==========
    print("\n=== 📊 标注框坐标对比 ===")
    if xyxy_list:
        print(f"原始标注框（xyxy像素）：{[round(x, 1) for x in xyxy_list[0]]}")
    if len(aug_xyxy) > 0:
        print(f"增强后标注框（xyxy像素）：{[round(x, 1) for x in aug_xyxy[0]]}")
    else:
        print("⚠️  增强后无有效标注框")

    print("\n🎉 波动增强完成！请查看保存的图片文件。")