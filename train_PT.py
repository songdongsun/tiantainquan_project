from tests.test_python import image

from ultralytics import YOLO
import torch




if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("运行设备：{}",device)

    # 2. 第一步：加载自定义分割配置（纯分割，无文本）
    model = YOLO("./tiantianquan/cfg/tiantianquan-pt.yaml")

    # 3. 第二步：加载官方预训练权重（分开调用，避免链式冲突）
    model.load("yolov8n.pt")

    # model = YOLO('yolov8n-seg.pt')

    print("模型加载成功！")
    model.info()  # 打印模型结构

    results = model.train(data="./tiantianquan/cfg/dataset_pt.yaml",
                          device=device,
                          epochs = 200,
                          patience = 15,
                          optimizer='SGD',  # 指定优化器为AdamW
                          lr0=0.0008,  # AdamW学习率（SGD的1/10）
                          weight_decay=0.0005,
                          cos_lr=True,
                          )

