from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO("tiantianquan/pt/segment/train7/best_seg.pt")  # load a custom-trained model

    # Export the model
    model.export(format="onnx")