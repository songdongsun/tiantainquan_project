import cv2
import numpy as np
import random
import os
import torch
from ultralytics.data.augment import Wave  # 导入新版官方风格的WaveAugment
from ultralytics.utils.instance import Instances

# ===================== YOLO标签解析工具（保留你的原有逻辑） =====================
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


# ===================== 测试代码（完全适配你提供的Instances类） =====================
if __name__ == "__main__":
    # ========== 1. 配置参数 ==========
    INPUT_IMAGE_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\pt_ng\images\train\1 (2161)_2.jpg"
    LABEL_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\pt_ng\labels\train\1 (2161)_2.txt"
    OUTPUT_IMAGE_PATH = r"E:\work_space\tiantainquan_project\tiantianquan\dataset\bd\wave_augmented.jpg"
    AMPLITUDE = 8.0
    FREQUENCY = 0.5
    DIRECTION = "horizontal"

    # ========== 2. 读取图片 ==========
    original_img = cv2.imread(INPUT_IMAGE_PATH)
    if original_img is None:
        raise ValueError(f"无法读取图片，请检查路径：{INPUT_IMAGE_PATH}")
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img_rgb.shape[:2]
    print(f"✅ 成功读取图片：尺寸 {w}x{h}")

    # ========== 3. 读取YOLO标签 ==========
    xyxy_list, cls_list = read_yolo_labels(LABEL_PATH, w, h)
    print(f"✅ 成功读取 {len(xyxy_list)} 个标注框")

    # ========== 4. 构造你提供的Instances对象（核心纠正点1） ==========
    # 关键：你提供的Instances初始化参数是bboxes（np.ndarray）+ bbox_format，无xyxy/cls参数
    if len(xyxy_list) > 0:
        # 转换为np.ndarray，格式指定为xyxy（匹配标签解析结果）
        bboxes_np = np.array(xyxy_list, dtype=np.float32)
        test_instances = Instances(
            bboxes=bboxes_np,          # 核心参数：bboxes（不是xyxy）
            bbox_format="xyxy",        # 指定格式为xyxy
            normalized=False,          # 标签是像素坐标，非归一化
            segments=None,
            keypoints=None
        )
        # 额外保存cls_list（你提供的Instances无cls属性，单独存储）
        test_instances.cls = np.array(cls_list, dtype=np.float32)  # 扩展添加cls属性
    else:
        # 空实例：bboxes传空数组
        test_instances = Instances(
            bboxes=np.empty((0, 4), dtype=np.float32),
            bbox_format="xyxy",
            normalized=False
        )
        test_instances.cls = np.array([], dtype=np.float32)

    # ========== 5. 构造官方格式的labels字典 ==========
    labels = {
        "img": original_img_rgb,
        "instances": test_instances
    }

    # ========== 6. 应用波动增强（新版官方接口） ==========
    wave_aug = Wave(wave_p=1.0, wave_amplitude=AMPLITUDE, wave_frequency=FREQUENCY, wave_direction=DIRECTION)
    augmented_labels = wave_aug(labels)  # 调用新版__call__接口

    # 提取增强后的结果
    aug_img_rgb = augmented_labels["img"]
    aug_instances = augmented_labels["instances"]

    # ========== 7. 绘制标注框并保存图片（核心纠正点2） ==========
    aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)

    # 关键：你提供的Instances通过bboxes属性获取标注框（不是xyxy）
    aug_xyxy = aug_instances.bboxes  # 直接读取bboxes（np.ndarray）
    aug_cls = getattr(aug_instances, "cls", np.array([]))  # 读取扩展的cls属性

    # 绘制增强后的标注框
    for i, box in enumerate(aug_xyxy):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(aug_img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    # 保存增强后的图片
    cv2.imwrite(OUTPUT_IMAGE_PATH, aug_img_bgr)
    print(f"✅ 增强后的图片已保存至：{os.path.abspath(OUTPUT_IMAGE_PATH)}")

    # ========== 8. 保存原始图（带标注框） ==========
    original_img_bgr = original_img.copy()
    for box in xyxy_list:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original_img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    original_save_path = os.path.join(os.path.dirname(OUTPUT_IMAGE_PATH), "original_with_box.jpg")
    cv2.imwrite(original_save_path, original_img_bgr)
    print(f"✅ 原始图片（带标注框）已保存至：{os.path.abspath(original_save_path)}")

    # ========== 9. 输出坐标对比 ==========
    print("\n=== 📊 标注框坐标对比 ===")
    if xyxy_list:
        print(f"原始标注框（xyxy像素）：{[round(x, 1) for x in xyxy_list[0]]}")
    if len(aug_xyxy) > 0:
        print(f"增强后标注框（xyxy像素）：{[round(x, 1) for x in aug_xyxy[0]]}")
    else:
        print("⚠️  增强后无有效标注框")

    print("\n🎉 波动增强完成！请查看保存的图片文件。")