import os
import math
import zipfile
from shutil import copy2


def split_and_zip_dataset(input_dir, output_root_dir, person_names):
    """
    核心功能：平均分配图片数据集，文件夹/压缩包名后追加图片数量，最后打包ZIP
    参数：
        input_dir: 原始图片数据集文件夹路径
        output_root_dir: 输出根文件夹（基础名，会追加总图片数）
        person_names: 人员名字列表
    """
    # 1. 基础校验
    if not os.path.exists(input_dir):
        raise ValueError(f"❌ 输入文件夹不存在：{input_dir}")
    if not person_names:
        raise ValueError(f"❌ 人员名单不能为空")

    # 2. 筛选支持的图片格式
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    total_imgs = len(img_files)
    total_persons = len(person_names)

    if total_imgs == 0:
        print("⚠️  输入文件夹中未找到支持的图片文件（png/jpg/jpeg/bmp/tiff/gif）")
        return

    # 3. 重命名输出根文件夹（追加总图片数）
    output_root_dir_with_count = f"{output_root_dir}_{total_imgs}张"
    os.makedirs(output_root_dir_with_count, exist_ok=True)

    # 4. 计算平均分配规则
    imgs_per_person = math.floor(total_imgs / total_persons)
    remainder = total_imgs % total_persons

    # 5. 遍历人员，分配图片并创建带数量的文件夹
    img_idx = 0
    person_img_count = {}  # 记录每个人实际分配的图片数
    for person_idx, name in enumerate(person_names):
        # 计算当前人员应分配的图片数
        current_count = imgs_per_person + 1 if person_idx < remainder else imgs_per_person
        person_img_count[name] = current_count

        # 创建带「人名+图片数」的文件夹
        person_dir = os.path.join(output_root_dir_with_count, f"{name}_{current_count}张")
        os.makedirs(person_dir, exist_ok=True)

        # 复制图片到对应文件夹
        for _ in range(current_count):
            if img_idx >= total_imgs:
                break

            img_name = img_files[img_idx]
            src_path = os.path.join(input_dir, img_name)
            dst_path = os.path.join(person_dir, img_name)

            copy2(src_path, dst_path)
            print(f"✅ {img_name} → {name}_{current_count}张 文件夹")

            img_idx += 1

    # 6. 打包为带「总图片数」的ZIP（保留文件夹结构）
    zip_filename = f"数据集分配结果_{total_imgs}张.zip"
    zip_path = os.path.join(output_root_dir_with_count, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_root_dir_with_count):
            if zip_filename in root:
                continue
            for file in files:
                file_full_path = os.path.join(root, file)
                zip_inner_path = os.path.relpath(file_full_path, output_root_dir_with_count)
                zipf.write(file_full_path, zip_inner_path)

    # 7. 输出详细统计
    print("\n" + "-" * 60)
    print(f"📊 分配统计结果：")
    print(f"   总图片数：{total_imgs}")
    print(f"   总人数：{total_persons}")
    print(f"   每人分配详情：")
    for name, count in person_img_count.items():
        print(f"     - {name}：{count} 张")
    print(f"\n📂 输出根文件夹：{output_root_dir_with_count}")
    print(f"📦 压缩包已生成：{zip_path}")


# ------------------- 只需修改以下参数 -------------------
if __name__ == "__main__":
    # 1. 替换为你的原始图片数据集文件夹路径
    INPUT_DATASET_DIR = r"data/dataset3/images"

    # 2. 替换为输出根文件夹（会自动创建）
    OUTPUT_ROOT_BASE = r"E:\workspace-pycharm\ttq_dataset\data\share"

    # 3. 替换为你的人员名字列表（可自由增删）
    PERSON_LIST = [
        "黄丽",
        "郭子昀",
        "赵鸿博",
        "范宇飞",
        "翁刚祥",
        "王卷卷",
        "吴海峰",
        "仰康佳",
        "钟敏",
        "纪晓曼",
        "丁慧芳",
        "陈镜明",
        "余佳科",
        "张璇",
        "唐雅楠",
        "田久杨",
        "彭嫣然",
        "段娅妮",
        "马恒"
    ]

    # 执行分配和打包
    split_and_zip_dataset(INPUT_DATASET_DIR, OUTPUT_ROOT_BASE, PERSON_LIST)