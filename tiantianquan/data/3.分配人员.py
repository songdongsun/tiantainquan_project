import os
import math
import zipfile
from shutil import copy2


def split_and_zip_dataset(input_dir, output_root_dir, person_names):
    """
    核心功能：将输入文件夹中的图片平均分配给指定人员，按名字建文件夹，最后打包为ZIP（保留文件夹结构）
    参数：
        input_dir: 原始图片数据集文件夹路径
        output_root_dir: 输出根文件夹（存放各人名文件夹+压缩包）
        person_names: 人员名字列表（用于创建文件夹和分配）
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

    # 3. 计算平均分配规则
    imgs_per_person = math.floor(total_imgs / total_persons)  # 每人基础数量
    remainder = total_imgs % total_persons  # 剩余需均分的数量（前remainder人多1张）

    # 4. 创建输出根文件夹
    os.makedirs(output_root_dir, exist_ok=True)

    # 5. 遍历人员，分配并复制图片
    img_idx = 0  # 图片遍历指针
    for person_idx, name in enumerate(person_names):
        # 创建当前人员的文件夹
        person_dir = os.path.join(output_root_dir, name)
        os.makedirs(person_dir, exist_ok=True)

        # 确定当前人员应分配的图片数（前remainder人多1张）
        current_count = imgs_per_person + 1 if person_idx < remainder else imgs_per_person

        # 复制图片到对应文件夹
        for _ in range(current_count):
            if img_idx >= total_imgs:
                break  # 图片分配完毕

            img_name = img_files[img_idx]
            src_path = os.path.join(input_dir, img_name)
            dst_path = os.path.join(person_dir, img_name)

            # 复制图片（保留原文件属性）
            copy2(src_path, dst_path)
            print(f"✅ {img_name} → {name} 文件夹")

            img_idx += 1

    # 6. 打包所有人员文件夹为ZIP（保留文件夹结构）
    zip_filename = "数据集分配结果.zip"
    zip_path = os.path.join(output_root_dir, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历输出目录，打包所有人员文件夹
        for root, dirs, files in os.walk(output_root_dir):
            # 跳过已生成的ZIP文件，避免递归打包
            if zip_filename in root:
                continue

            for file in files:
                file_full_path = os.path.join(root, file)
                # 计算ZIP内的相对路径（保留「人名/图片名」结构）
                zip_inner_path = os.path.relpath(file_full_path, output_root_dir)
                zipf.write(file_full_path, zip_inner_path)

    # 7. 输出分配统计
    print("\n" + "-" * 50)
    print(f"📊 分配统计结果：")
    print(f"   总图片数：{total_imgs}")
    print(f"   总人数：{total_persons}")
    print(f"   每人基础分配数：{imgs_per_person} 张")
    if remainder > 0:
        print(f"   前 {remainder} 人各多分配1张")
    print(f"\n📦 压缩包已生成：{zip_path}")
    print(f"📂 各人员文件夹路径：{output_root_dir}")


# ------------------- 只需修改以下参数 -------------------
if __name__ == "__main__":
    # 1. 替换为你的原始图片数据集文件夹路径
    INPUT_DATASET_DIR = r"data/dataset3/images"

    # 2. 替换为输出根文件夹（会自动创建）
    OUTPUT_ROOT_DIR = r"E:\workspace-pycharm\ttq_dataset\data\share"

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
    split_and_zip_dataset(INPUT_DATASET_DIR, OUTPUT_ROOT_DIR, PERSON_LIST)