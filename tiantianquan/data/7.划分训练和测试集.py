
from torch.utils.data import DataLoader, Dataset  # 生成器
import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path




class MyDataSet(Dataset):
    def __init__(self, root, split="CX", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # 遍历类别文件夹
        for cls_idx, cls_name in enumerate(os.listdir(os.path.join(root, split))):
            cls_dir = os.path.join(root, split, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".png", ".bmp")):
                    self.img_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

    #划分训练和测试数据集
    def split_image_dataset(self, raw_data_dir, output_dir, test_ratio=0.1, val_ratio=0.1, random_seed=2025):
        """
        分层划分图片数据集为 train/val/test 三个子集（按类别保证分布一致）
        :param raw_data_dir: 原始数据根目录（按类别分文件夹）
        :param output_dir: 划分后数据保存目录
        :param test_ratio: 测试集占总数据的比例（默认0.1，即10%）
        :param val_ratio: 验证集占「训练+验证集」的比例（默认0.1，即从90%的train_val中再分10%作为val）
                          最终比例：train ≈ (1-test_ratio)*(1-val_ratio)，val≈(1-test_ratio)*val_ratio，test≈test_ratio
                          示例：test_ratio=0.1, val_ratio=1/9 → train:val:test = 8:1:1
        :param random_seed: 随机种子（保证划分可复现）
        """
        # 1. 设置随机种子（保证划分结果可复现）
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 2. 创建输出目录（新增val子目录）
        for split in ["train", "val", "test"]:
            split_dir = Path(output_dir) / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"创建目录：{split_dir}")

        # 3. 遍历所有类别文件夹（分层划分的核心：按类别单独处理）
        class_folders = [f for f in os.listdir(raw_data_dir)
                         if os.path.isdir(os.path.join(raw_data_dir, f))]

        if not class_folders:
            raise ValueError("原始数据目录下没有找到类别文件夹，请检查目录路径！")

        print(f"\n===== 数据集划分开始 =====")
        print(f"发现 {len(class_folders)} 个类别：{class_folders}")
        print(
            f"划分比例：测试集 {test_ratio * 100:.1f}% | 验证集 {(1 - test_ratio) * val_ratio * 100:.1f}% | 训练集 {(1 - test_ratio) * (1 - val_ratio) * 100:.1f}%")

        # 4. 按类别逐一分层划分
        total_train, total_val, total_test = 0, 0, 0  # 统计全局数量
        for class_name in class_folders:
            # 拼接当前类别的目录路径
            class_dir = Path(raw_data_dir) / class_name
            # 过滤并获取该类别下所有图片文件
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(image_extensions)]

            if len(image_files) == 0:
                print(f"\n警告：类别 {class_name} 下没有找到图片文件，跳过该类别！")
                continue

            class_total = len(image_files)
            print(f"\n处理类别 {class_name}：共 {class_total} 张图片")

            # ===== 第一步：划分 训练+验证集(train_val) 和 测试集(test) =====
            train_val_files, test_files = train_test_split(
                image_files,
                test_size=test_ratio,
                random_state=random_seed,
                stratify=[class_name] * class_total  # 分层保证类别分布（单类别下等价于随机，但扩展多类别更鲁棒）
            )

            # ===== 第二步：从 train_val 中划分 训练集(train) 和 验证集(val) =====
            train_files, val_files = train_test_split(
                train_val_files,
                test_size=val_ratio,
                random_state=random_seed,
                stratify=[class_name] * len(train_val_files)
            )

            # 5. 定义内部函数：复制文件到指定目录（避免重复代码）
            def copy_files(file_list, split):
                dest_dir = Path(output_dir) / split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for file_name in file_list:
                    src_path = class_dir / file_name
                    dest_path = dest_dir / file_name
                    try:
                        shutil.copy2(src_path, dest_path)  # 保留文件元信息（创建时间、权限等）
                    except Exception as e:
                        print(f"  复制文件失败 {file_name}：{e}")

            # 6. 复制文件到对应目录（新增val）
            copy_files(train_files, "train")
            copy_files(val_files, "val")
            copy_files(test_files, "test")

            # 7. 打印当前类别的划分结果
            train_num, val_num, test_num = len(train_files), len(val_files), len(test_files)
            print(f"  - 训练集：{train_num} 张 ({train_num / class_total * 100:.1f}%)")
            print(f"  - 验证集：{val_num} 张 ({val_num / class_total * 100:.1f}%)")
            print(f"  - 测试集：{test_num} 张 ({test_num / class_total * 100:.1f}%)")

            # 累加全局数量
            total_train += train_num
            total_val += val_num
            total_test += test_num

        # 8. 打印全局划分结果
        total_all = total_train + total_val + total_test
        print(f"\n===== 数据集划分完成 ======")
        print(f"全局统计：")
        print(f"  总样本数：{total_all}")
        print(f"  训练集：{total_train} 张 ({total_train / total_all * 100:.1f}%)")
        print(f"  验证集：{total_val} 张 ({total_val / total_all * 100:.1f}%)")
        print(f"  测试集：{total_test} 张 ({total_test / total_all * 100:.1f}%)")
        print(f"划分后数据保存路径：{os.path.abspath(output_dir)}")
    #过采样
    def oversample_small_class(self,raw_data_dir, target_class, target_num, save_dir, random_seed=2025):
        """
        对小类进行过采样（带数据增强）
        :param raw_data_dir: 原始数据根目录
        :param target_class: 小类名称（如xh）
        :param target_num: 目标样本数（如200）
        :param save_dir: 增强后保存目录
        """
        np.random.seed(random_seed)
        class_dir = Path(raw_data_dir) / target_class
        save_class_dir = Path(save_dir) / target_class
        save_class_dir.mkdir(parents=True, exist_ok=True)

        # 获取小类原始样本
        image_extensions = (".jpg", ".png", ".bmp")
        raw_files = [f for f in os.listdir(class_dir) if f.lower().endswith(image_extensions)]
        raw_num = len(raw_files)
        if raw_num >= target_num:
            print(f"类别{target_class}样本数已达标，无需过采样")
            return

        # 计算需要生成的样本数
        need_num = target_num - raw_num
        print(f"对类别{target_class}过采样：原始{raw_num}张 → 目标{target_num}张（需生成{need_num}张）")

        # 对原始样本循环增强，直到达到目标数
        augment_idx = 0
        while augment_idx < need_num:
            for raw_file in raw_files:
                if augment_idx >= need_num:
                    break
                # 读取图片
                img_path = class_dir / raw_file
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 随机数据增强（和训练集增强一致，避免分布不一致）
                # 1. 随机水平翻转
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                # 2. 随机旋转（-15~15度）
                angle = np.random.randint(-15, 16)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
                # # 3. 随机亮度调整
                # if np.random.rand() > 0.5:
                #     img = img * (0.8 + np.random.rand() * 0.4)
                #     img = np.clip(img, 0, 255).astype(np.uint8)

                # 保存增强后的样本
                save_name = f"{raw_file.split('.')[0]}_aug_{augment_idx}.png"
                save_path = save_class_dir / save_name
                cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                augment_idx += 1
    #欠采样
    def undersample_large_class(self,raw_data_dir, target_class, target_num, save_dir, random_seed=2025):
        """对大类欠采样"""
        np.random.seed(random_seed)
        class_dir = Path(raw_data_dir) / target_class
        save_class_dir = Path(save_dir) / target_class
        save_class_dir.mkdir(parents=True, exist_ok=True)

        raw_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".bmp"))]
        raw_num = len(raw_files)
        if raw_num <= target_num:
            print(f"类别{target_class}样本数不足，无需欠采样")
            return

        # 随机选择target_num个样本
        selected_files = np.random.choice(raw_files, size=target_num, replace=False)
        print(f"对类别{target_class}欠采样：原始{raw_num}张 → 目标{target_num}张")

        # 复制选中的样本
        for file in selected_files:
            src_path = class_dir / file
            dest_path = save_class_dir / file
            shutil.copy2(src_path, dest_path)





if __name__ == '__main__':
    dataset = MyDataSet('../images')
    # dataset.split_image_dataset(r'../images',r'../data2',0.1,0.2)

    # 调用示例：对xh（50张）过采样到200张，pw（88张）过采样到300张
    # dataset.oversample_small_class(
    #     raw_data_dir="E:\workspace-pycharm\hanpan-pro\data2\\train",
    #     target_class="WHD",
    #     target_num=500,
    #     save_dir="../data2_balanced/train"
    # )
    # dataset.oversample_small_class(
    #     raw_data_dir="E:\workspace-pycharm\hanpan-pro\data2\\val",
    #     target_class="WHD",
    #     target_num=130,
    #     save_dir="../data2_balanced/val"
    # )
    # 大类（cc、cx）无需处理，直接复制到balanced目录

    # 调用示例：cc降到1000张，cx降到500张
    # dataset.undersample_large_class(
    #     raw_data_dir="../images",
    #     target_class="OK",
    #     target_num=1000,
    #     save_dir="../images_balanced"
    # )
