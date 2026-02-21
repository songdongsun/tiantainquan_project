import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ===================== 配置参数（根据你的数据集修改） =====================
# YOLO格式数据集根目录（包含images/和labels/两个子文件夹）
DATA_ROOT = r"E:\workspace-pycharm\ultralytics\tiantianquan\dataset\target_data\ng"
# 划分比例（训练集:验证集）
TRAIN_RATIO = 0.8
# 随机种子（固定种子保证划分结果可复现）
SEED = 42
# 是否创建软链接（True=不复制文件，节省空间；False=复制文件）
USE_SYMLINK = False


# ===================== 核心函数 =====================
def load_yolo_samples(data_root):
    """
    加载YOLO数据集，按类别分组样本
    返回：
        class2samples: 字典 {类别ID: [样本名列表]}
        all_samples: 所有样本名集合（无后缀）
    """
    # 定义路径
    images_dir = Path(data_root) / "images"
    labels_dir = Path(data_root) / "labels"
    assert images_dir.exists(), f"图片文件夹 {images_dir} 不存在！"
    assert labels_dir.exists(), f"标注文件夹 {labels_dir} 不存在！"

    # 按类别分组样本
    class2samples = defaultdict(list)
    all_samples = set()

    # 遍历所有标注文件
    for label_file in labels_dir.glob("*.txt"):
        sample_name = label_file.stem  # 样本名（无后缀）
        all_samples.add(sample_name)

        # 读取标注文件，获取样本包含的类别
        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    class_id = line.strip().split()[0]  # YOLO格式：类别ID 中心x 中心y 宽 高
                    class2samples[class_id].append(sample_name)
                    # 一个样本可能包含多个类别，只记录一次即可（避免重复划分）
                    break

    # 去重（同一个样本可能出现在多个类别列表中）
    for cls in class2samples:
        class2samples[cls] = list(set(class2samples[cls]))

    print(f"共加载 {len(all_samples)} 个样本，涉及 {len(class2samples)} 个类别：{sorted(class2samples.keys())}")
    return class2samples, all_samples


def split_dataset(class2samples, all_samples, train_ratio=0.8, seed=42):
    """分层划分数据集，保证每个类别按比例拆分，并返回每个类别的train/val数量"""
    random.seed(seed)
    train_samples = set()
    val_samples = set()
    # 新增：记录每个类别的train/val数量
    class_split_info = defaultdict(dict)

    # 对每个类别单独划分
    for cls, samples in class2samples.items():
        random.shuffle(samples)  # 随机打乱
        split_idx = int(len(samples) * train_ratio)
        train_cls = samples[:split_idx]
        val_cls = samples[split_idx:]

        # 记录该类别的数量
        class_split_info[cls]["total"] = len(samples)
        class_split_info[cls]["train"] = len(train_cls)
        class_split_info[cls]["val"] = len(val_cls)
        class_split_info[cls]["train_ratio"] = round(len(train_cls) / len(samples), 4) if len(samples) > 0 else 0

        # 添加到全局集合
        train_samples.update(train_cls)
        val_samples.update(val_cls)

    # 打印每个类别的详细数量（核心新增功能）
    print("\n========== 各类型别划分详情 ==========")
    print(f"{'类别ID':<8} {'总数量':<8} {'训练集数量':<10} {'验证集数量':<10} {'训练集比例':<10}")
    print("-" * 50)
    for cls in sorted(class_split_info.keys()):
        info = class_split_info[cls]
        print(f"{cls:<8} {info['total']:<8} {info['train']:<10} {info['val']:<10} {info['train_ratio']:<10}")
    print("-" * 50)

    # 处理无标注的样本（如果有）
    unlabeled_samples = all_samples - (train_samples | val_samples)
    if unlabeled_samples:
        print(f"\n发现 {len(unlabeled_samples)} 个无标注样本，随机划分到训练集")
        train_samples.update(unlabeled_samples)

    # 最终校验
    assert len(train_samples & val_samples) == 0, "训练集和验证集存在重复样本！"
    print(f"\n========== 整体划分结果 ==========")
    print(f"训练集：{len(train_samples)} 个样本")
    print(f"验证集：{len(val_samples)} 个样本")
    print(f"总样本：{len(train_samples) + len(val_samples)}（原始：{len(all_samples)}）")
    print(f"整体训练集比例：{round(len(train_samples) / (len(train_samples) + len(val_samples)), 4)}")

    return train_samples, val_samples, class_split_info


def create_dataset_structure(data_root, train_samples, val_samples, use_symlink=True):
    """创建YOLO格式的数据集目录，并复制/链接文件"""
    root = Path(data_root)
    # 定义目标目录
    dirs = {
        "train_images": root / "train" / "images",
        "train_labels": root / "train" / "labels",
        "val_images": root / "val" / "images",
        "val_labels": root / "val" / "labels",
    }
    # 创建目录
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 复制/链接文件的函数
    def copy_or_link(src, dst):
        if use_symlink:
            # 创建软链接（Windows需管理员权限，Linux/Mac直接用）
            if not dst.exists():
                if os.name == "nt":  # Windows
                    os.symlink(src, dst, target_is_directory=False)
                else:  # Linux/Mac
                    dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)

    # 处理训练集
    for sample in train_samples:
        # 图片文件（兼容jpg/png等格式）
        img_srcs = list((root / "images").glob(f"{sample}.*"))
        if img_srcs:
            img_src = img_srcs[0]
            img_dst = dirs["train_images"] / img_src.name
            copy_or_link(img_src, img_dst)
        # 标注文件
        label_src = root / "labels" / f"{sample}.txt"
        if label_src.exists():
            label_dst = dirs["train_labels"] / label_src.name
            copy_or_link(label_src, label_dst)

    # 处理验证集
    for sample in val_samples:
        # 图片文件
        img_srcs = list((root / "images").glob(f"{sample}.*"))
        if img_srcs:
            img_src = img_srcs[0]
            img_dst = dirs["val_images"] / img_src.name
            copy_or_link(img_src, img_dst)
        # 标注文件
        label_src = root / "labels" / f"{sample}.txt"
        if label_src.exists():
            label_dst = dirs["val_labels"] / label_src.name
            copy_or_link(label_src, label_dst)

    # 生成train.txt/val.txt（记录样本绝对路径，适配YOLO训练）
    def write_sample_list(samples, save_path, img_dir):
        with open(save_path, "w", encoding="utf-8") as f:
            for sample in samples:
                img_srcs = list(img_dir.glob(f"{sample}.*"))
                if img_srcs:
                    f.write(f"{img_srcs[0].absolute()}\n")

    write_sample_list(train_samples, root / "train.txt", dirs["train_images"])
    write_sample_list(val_samples, root / "val.txt", dirs["val_images"])

    print(f"\n========== 目录创建完成 ==========")
    print(f"训练集图片：{dirs['train_images']}")
    print(f"训练集标注：{dirs['train_labels']}")
    print(f"验证集图片：{dirs['val_images']}")
    print(f"验证集标注：{dirs['val_labels']}")
    print(f"训练集列表：{root / 'train.txt'}")
    print(f"验证集列表：{root / 'val.txt'}")


# ===================== 执行划分 =====================
if __name__ == "__main__":
    # 1. 加载样本
    class2samples, all_samples = load_yolo_samples(DATA_ROOT)
    # 2. 分层划分（新增返回class_split_info）
    train_samples, val_samples, class_split_info = split_dataset(class2samples, all_samples, TRAIN_RATIO, SEED)
    # 3. 创建目录和文件
    create_dataset_structure(DATA_ROOT, train_samples, val_samples, USE_SYMLINK)