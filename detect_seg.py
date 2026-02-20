import cv2
from tiantianquan.detect.pt_onnx import SegPipeline


if __name__ == "__main__":
    # path = r"test_pt.jpg"
    path = r"./tiantianquan/dataset/seg_data/images/train/00deff74256fa17d7d7976af9eaa4261.png"
    image = cv2.imread(path)

    pipeline = SegPipeline(seg_model_path="./tiantianquan/pt/segment/train7/best_seg.onnx",
                          is_debug=True)

    results = pipeline.run(image)
    print("最终检测结果:", results)