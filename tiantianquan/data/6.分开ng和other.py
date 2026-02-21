import os
import shutil
from pathlib import Path


def classify_images_by_labels(
        source_root: str,
        output_root: str,
        image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        label_extension: str = '.txt'
):
    """
    基于 SOURCE_ROOT/images 和 SOURCE_ROOT/labels 文件夹分类数据，
    输出目录保持 images/labels 子文件夹结构：
    - pt_ng/images: 有标签的图片 | pt_ng/labels: 对应的标签文件
    - other/images: 无标签的图片 | other/labels: 不创建（无标签）
    （复制而非移动，不修改原数据）

    Args:
        source_root: 原始数据根目录（该目录下有images和labels子文件夹）
        output_root: 输出根目录（会自动创建ng/other及子文件夹）
        image_extensions: 支持的图片后缀（小写）
        label_extension: 标签文件后缀（小写）
    """
    # 转换为Path对象，明确原始路径
    source_path = Path(source_root)
    src_images = source_path / "images"  # 原始图片文件夹
    src_labels = source_path / "labels"  # 原始标签文件夹
    output_path = Path(output_root)

    # 检查原始目录是否存在
    if not source_path.exists():
        raise FileNotFoundError(f"原始数据根目录不存在: {source_root}")
    if not src_images.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {src_images}")
    if not src_labels.exists():
        raise FileNotFoundError(f"标签文件夹不存在: {src_labels}")

    # ========== 1. 创建输出目录结构 ==========
    # NG目录结构：pt_ng/images + pt_ng/labels
    ng_root = output_path / "pt_ng"
    ng_images = ng_root / "images"
    ng_labels = ng_root / "labels"
    ng_images.mkdir(parents=True, exist_ok=True)
    ng_labels.mkdir(parents=True, exist_ok=True)

    # OTHER目录结构：仅创建other/images（无labels）
    other_root = output_path / "other"
    other_images = other_root / "images"
    other_images.mkdir(parents=True, exist_ok=True)

    # 统计变量
    total_image_count = 0
    ng_image_count = 0
    ng_label_count = 0
    other_image_count = 0

    # 遍历原始images文件夹中的所有图片
    image_files = [f for f in src_images.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    # 开始处理
    print("===== 开始处理图片 =====")
    for image_file in image_files:
        total_image_count += 1
        img_name = image_file.name
        # 构建对应的标签文件路径
        label_filename = image_file.stem + label_extension
        label_filepath = src_labels / label_filename

        if label_filepath.exists():
            # ========== 有标签 → NG目录（保持images/labels结构） ==========
            # 复制图片到 pt_ng/images
            img_target = ng_images / img_name
            shutil.copy2(image_file, img_target)
            ng_image_count += 1

            # 复制标签到 pt_ng/labels
            label_target = ng_labels / label_filename
            shutil.copy2(label_filepath, label_target)
            ng_label_count += 1

            # 打印处理信息
            print(f"[NG] 处理图片: {img_name} → pt_ng/images")
            print(f"     对应标签: {label_filename} → pt_ng/labels")
        else:
            # ========== 无标签 → OTHER目录（仅images） ==========
            # 复制图片到 other/images
            img_target = other_images / img_name
            shutil.copy2(image_file, img_target)
            other_image_count += 1

            # 打印处理信息
            print(f"[OTHER] 处理图片: {img_name} → other/images (无对应标签)")

    # ========== 输出统计结果 ==========
    print("\n===== 分类完成 =====")
    print(f"原始images文件夹总图片数: {total_image_count}")
    print("-" * 40)
    print(f"NG目录结构: {ng_root}")
    print(f"  - pt_ng/images: {ng_image_count} 张图片")
    print(f"  - pt_ng/labels: {ng_label_count} 个标签文件")
    print("-" * 40)
    print(f"OTHER目录结构: {other_root}")
    print(f"  - other/images: {other_image_count} 张图片")
    print(f"  - other/labels: 无（无对应标签）")


# ==================== 配置参数 ====================
if __name__ == "__main__":
    # 原始数据根目录（含images/labels子文件夹）
    SOURCE_ROOT = r"E:\workspace-pycharm\ultralytics\tiantianquan\dataset\spilt_data"
    # 输出根目录（会生成ng/other及子文件夹）
    OUTPUT_ROOT_DIR = r"E:\workspace-pycharm\ultralytics\tiantianquan\dataset\target_data"

    # 执行分类
    classify_images_by_labels(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT_DIR
    )