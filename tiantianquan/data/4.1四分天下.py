import cv2
import os
from typing import List, Tuple


def yolo2pix(yolo_str: str, img_h: int, img_w: int) -> Tuple[int, int, int, int, int]:
    """YOLO归一化坐标 → 像素坐标（class_id, x1, y1, x2, y2）"""
    str_list = yolo_str.strip().split()
    class_id = int(str_list[0])
    x_c, y_c, box_w, box_h = map(float, str_list[1:len(str_list)])

    x1 = (x_c - box_w / 2) * img_w
    y1 = (y_c - box_h / 2) * img_h
    x2 = (x_c + box_w / 2) * img_w
    y2 = (y_c + box_h / 2) * img_h
    return class_id, int(x1), int(y1), int(x2), int(y2)


def pix2yolo(pix_box: Tuple[int, int, int, int], img_h: int, img_w: int) -> Tuple[float, float, float, float]:
    """像素坐标（x1,y1,x2,y2）→ YOLO归一化坐标（cx, cy, w, h）"""
    x1, y1, x2, y2 = pix_box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # 限制坐标在0~1之间，避免浮点误差
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def get_crop_areas(img_h: int, img_w: int, area_size: int) -> List[Tuple[int, int, int, int]]:
    """计算4个裁剪区域坐标（以中心为基准，向4个方向偏移）"""
    img_w_c = int(img_w / 2)
    img_h_c = int(img_h / 2)
    # 4个区域：左上(0)、右上(1)、左下(2)、右下(3)（x_min, y_min, x_max, y_max）
    cut_areas = [
        (min(0, img_w_c), min(0, img_h_c), max(img_w_c, img_w_c + area_size), max(img_h_c, img_h_c + area_size)),
        (min(img_w_c, img_w_c - area_size), min(img_h_c, 0), max(img_w_c, img_w), max(img_h_c, img_h_c + area_size)),
        (min(img_w_c, 0), min(img_h_c, img_h_c - area_size), max(img_w_c, img_w_c + area_size), max(img_h_c, img_h)),
        (min(img_w_c, img_w_c - area_size), min(img_h_c, img_h_c - area_size), max(img_w, img_w_c), max(img_h, img_h_c))
    ]
    return cut_areas


def filter_small_box(box: Tuple[int, int, int, int], min_size: int = 15) -> bool:
    """
    过滤小框：判断框的宽/高是否都≥最小尺寸（默认15px）
    :param box: (x1, y1, x2, y2) 像素坐标框
    :param min_size: 最小尺寸（px）
    :return: True=保留，False=删除
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    return box_w >= min_size and box_h >= min_size


def process_single_image(img_path: str, label_path: str, output_images_dir: str, output_labels_dir: str,
                         area_size: int):
    """处理单张图片：裁剪4个子图 + 生成对应YOLO标签（过滤15×15px小框，无标签不生成文件）"""
    # 1. 读取图片和基本信息
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  图片读取失败：{img_path}")
        return
    img_h, img_w = img.shape[:2]
    img_basename = os.path.splitext(os.path.basename(img_path))[0]  # 原文件名（无后缀）
    img_ext = os.path.splitext(os.path.basename(img_path))[1]  # 图片后缀（.jpg/.png）

    # 2. 读取YOLO标签（支持多行标签，多目标）
    yolo_lines = []
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as fr:
            yolo_lines = [line.strip() for line in fr if line.strip()]
    if not yolo_lines:
        print(f"⚠️  原图片无标签：{img_basename}，仅保存裁剪子图，不生成标签")

    # 3. 计算4个裁剪区域
    cut_areas = get_crop_areas(img_h, img_w, area_size)
    area_suffix = ["0", "1", "2", "3"]  # 左上_0、右上_1、左下_2、右下_3

    # 4. 遍历每个裁剪区域，生成子图和新标签
    for idx, (cut_area, suffix) in enumerate(zip(cut_areas, area_suffix)):
        x_min, y_min, x_max, y_max = cut_area
        # 裁剪子图
        son_img = img[y_min:y_max, x_min:x_max]
        son_h, son_w = son_img.shape[:2]
        if son_h == 0 or son_w == 0:
            print(f"⚠️  {img_basename}_{suffix}：裁剪出空图，跳过")
            continue

        # 5. 生成子图对应的YOLO标签（过滤15×15px小框）
        new_yolo_lines = []
        for yolo_line in yolo_lines:
            # YOLO→像素（原图坐标）
            class_id, ob_x1, ob_y1, ob_x2, ob_y2 = yolo2pix(yolo_line, img_h, img_w)
            # 判断是否有交集（无交集则跳过该目标）
            if ob_x2 <= x_min or ob_x1 >= x_max or ob_y2 <= y_min or ob_y1 >= y_max:
                continue

            # 计算交集区域（被裁切后的小框，原图坐标）
            inter_x1 = max(ob_x1, x_min)
            inter_y1 = max(ob_y1, y_min)
            inter_x2 = min(ob_x2, x_max)
            inter_y2 = min(ob_y2, y_max)

            # 过滤小框：＜15×15px则删除，不保留该框
            if not filter_small_box((inter_x1, inter_y1, inter_x2, inter_y2), min_size=15):
                print(f"ℹ️  {img_basename}_{suffix}：目标框尺寸＜15×15px，删除该标签")
                continue

            # 转换为子图相对坐标
            new_x1 = inter_x1 - x_min
            new_y1 = inter_y1 - y_min
            new_x2 = inter_x2 - x_min
            new_y2 = inter_y2 - y_min

            # 像素→YOLO（基于子图尺寸）
            cx, cy, w, h = pix2yolo((new_x1, new_y1, new_x2, new_y2), son_h, son_w)
            new_yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 6. 保存子图（必保存）
        son_img_name = f"{img_basename}_{suffix}{img_ext}"  # 命名：原名称_0.jpg
        son_img_path = os.path.join(output_images_dir, son_img_name)
        save_success = cv2.imwrite(son_img_path, son_img)
        if save_success:
            print(f"✅ 子图保存成功：{son_img_path}")
        else:
            print(f"❌ 子图保存失败：{son_img_path}")
            continue  # 子图保存失败则跳过标签保存

        # 7. 保存标签（仅当有有效标签时保存，无标签则不生成文件）
        if new_yolo_lines:
            son_label_name = f"{img_basename}_{suffix}.txt"  # 命名：原名称_0.txt
            son_label_path = os.path.join(output_labels_dir, son_label_name)
            with open(son_label_path, 'w', encoding='utf-8') as fw:
                fw.write("\n".join(new_yolo_lines))
            print(f"✅ 标签保存成功：{son_label_path}")
        else:
            print(f"ℹ️  {img_basename}_{suffix}：无有效标签，不生成标签文件")


def batch_process_dataset(input_root: str, output_root: str, area_size: int = 250):
    """
    批量处理YOLO数据集
    :param input_root: 输入根目录（含images/labels子目录）
    :param output_root: 输出根目录（自动创建images/labels子目录）
    :param area_size: 裁剪偏移尺寸（像素）
    """
    # 1. 校验输入目录
    input_images = os.path.join(input_root, "images")
    input_labels = os.path.join(input_root, "labels")
    for dir_path in [input_images, input_labels]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"输入目录不存在：{dir_path}")

    # 2. 创建输出目录（统一的images/labels）
    output_images = os.path.join(output_root, "images")
    output_labels = os.path.join(output_root, "labels")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # 3. 遍历所有图片
    supported_formats = ("jpg", "png", "jpeg", "bmp")
    total_img_count = 0
    for img_name in os.listdir(input_images):
        img_ext = img_name.split(".")[-1].lower()
        if img_ext not in supported_formats:
            continue

        # 匹配图片和标签路径
        img_basename = os.path.splitext(img_name)[0]
        img_path = os.path.join(input_images, img_name)
        label_path = os.path.join(input_labels, f"{img_basename}.txt")

        # 处理单张图片
        process_single_image(img_path, label_path, output_images, output_labels, area_size)
        total_img_count += 1

    print(f"\n📊 批量处理完成！")
    print(f"总处理原图数：{total_img_count}")
    print(f"📂 输出目录结构：")
    print(f"  {output_root}/")
    print(f"    ├─ images/  # 所有裁剪子图（原名称_0/1/2/3.jpg）")
    print(f"    └─ labels/  # 仅含有效标签的文件（原名称_0/1/2/3.txt）")


if __name__ == '__main__':
    # ====================== 只需配置以下参数 ======================
    INPUT_DATASET_ROOT = r"E:\workspace-pycharm\ultralytics\tiantianquan\dataset\src_data"  # 输入数据集根目录
    OUTPUT_DATASET_ROOT = r"E:\workspace-pycharm\ultralytics\tiantianquan\dataset\spilt_data"  # 输出根目录
    CROP_AREA_SIZE = 250  # 裁剪偏移尺寸（像素，匹配你说的250px重叠区域）
    # ============================================================

    try:
        batch_process_dataset(INPUT_DATASET_ROOT, OUTPUT_DATASET_ROOT, CROP_AREA_SIZE)
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")