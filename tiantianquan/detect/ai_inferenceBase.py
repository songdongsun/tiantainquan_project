import onnxruntime as ort
from typing import Optional, Tuple, List, Dict, Union, Any
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor


def show(img, title='Image', is_destroy=False):
    """显示图像"""
    if img is None or img.size == 0:
        print(f"无法显示图像: {title}")
        return

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 400, 400)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    if is_destroy: cv2.destroyAllWindows()
class AIInferenceBase:
    """
    AI推理基类（支持分类/检测/分割/关键点等任务的公共流程）
    任务类型: classification/detection/segmentation/keypoint
    """

    def __init__(
            self,
            model_path: str,
            task_type: str = "detection",
            input_size: Tuple[int, int] = (640, 640),  # (H, W)
            use_gpu: bool = False,
            yolo_type: str = "yolov8",
            **kwargs
    ):
        """
        :param task_type: 任务类型
        :param input_size: 模型输入尺寸 (高度, 宽度)
        :param use_gpu: 是否使用GPU推理
        :param kwargs: 任务特定参数（如类别名称、置信度阈值等）
        """
        self.task_type = task_type
        self.input_size = input_size
        self.use_gpu = use_gpu
        self.class_names = kwargs.get("class_names", [])
        self.yolo_type = yolo_type

        # ONNX Runtime初始化
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        # 会话周期。ort类的基础对象
        # 加载onnx模型结构到内存中。模型只需要加载1次。
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 必须，动态去获取，输入或者输出节点的名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"✅ 初始化成功 | 任务: {task_type} | 设备: {'GPU' if use_gpu else 'CPU'} | 输入尺寸: {input_size}")

    # ------------------------- 预处理方法 -------------------------
    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        raise NotImplementedError("子类需实现resize逻辑")

    def preprocess_normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化图像"""
        raise NotImplementedError("子类需实现归一化逻辑")

    def preprocess_to_tensor(self, image: np.ndarray) -> np.ndarray:
        """转换为NCHW张量"""

        return np.transpose(image, (2, 0, 1))[np.newaxis, ...]

    def run_preprocess(self, image: np.ndarray) -> np.ndarray:
        """完整预处理流水线 pipline"""
        image = self.preprocess_resize(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess_normalize(image)
        return self.preprocess_to_tensor(image)

    # ------------------------- 核心推理 -------------------------
    def forward(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """执行模型推理"""
        # 从onnx 中去找输入和输出节点的名称。固定的
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    # ------------------------- 后处理方法 -------------------------
    def postprocess_classification(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        """分类后处理"""
        raise NotImplementedError

    def postprocess_detection(self, outputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """检测后处理"""
        raise NotImplementedError

    def postprocess_yolo_segmentation(self, outputs: List[np.ndarray]) -> np.ndarray:
        """分割后处理（统一接口）"""
        raise NotImplementedError

    def postprocess_mask_segmentation(self, outputs: List[np.ndarray]) -> np.ndarray:
        """分割后处理（统一接口）"""
        raise NotImplementedError

    def postprocess_keypoint(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        """关键点后处理"""
        raise NotImplementedError

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    # ------------------------- 统一接口 -------------------------
    def predict(self, image: np.ndarray) -> Any:
        """通用预测接口"""
        tensor = self.run_preprocess(image)
        tensor = tensor.astype(np.float32)
        outputs = self.forward(tensor)

        if self.task_type == "classification":
            return self.postprocess_classification(outputs)
        elif self.task_type == "detection":
            return self.postprocess_detection(outputs)
        elif self.task_type == "yolo-segmentation":
            return self.postprocess_yolo_segmentation(outputs)
        elif self.task_type == "mask-segmentation":
            return self.postprocess_mask_segmentation(outputs)
        elif self.task_type == "keypoint":
            return self.postprocess_keypoint(outputs)
        raise ValueError(f"未知任务类型: {self.task_type}")


class ImageClassifier(AIInferenceBase):
    """图像分类任务实现"""

    def __init__(self, model_path: str, class_names: Optional[List[str]] = None, use_gpu: bool = False):
        super().__init__(
            model_path=model_path,
            task_type="classification",
            input_size=(224, 224),
            use_gpu=use_gpu,
            class_names=class_names or []
        )

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.input_size[1], self.input_size[0]))

    def preprocess_normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image - mean) / std

    def postprocess_classification(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        prob = self.softmax(outputs[0])[0]
        class_id = np.argmax(prob)
        return {
            "class": self.class_names[class_id] if self.class_names else str(class_id),
            "confidence": float(prob[class_id]),
            "all_probs": prob.tolist() if len(self.class_names) == len(prob) else None
        }


class ImageYoloDetect(AIInferenceBase):
    """YOLO目标检测任务实现（支持v5/v8等版本）"""

    def __init__(self,
                 model_path: str,
                 class_names: Optional[List[str]] = None,
                 use_gpu: bool = False,
                 confidence_thres: float = 0.25,
                 nms_thres: float = 0.45,
                 num_threads: int = 5,
                 distance_threshold: int = 30,
                 task_type="detection",
                 min_box_hw: Tuple[int, int] = (10, 10),  # 修改为Tuple类型
                 model_size = (640, 640),
                 yolo_type: str = "yolov8"):
        """
        :param min_box_hw: 最小检测框尺寸 (height, width)
        """
        super().__init__(
            model_path=model_path,
            task_type=task_type,
            input_size=model_size,
            use_gpu=use_gpu,
            class_names=class_names or []
        )
        self.num_threads = num_threads
        self.distance_threshold = distance_threshold
        # 检测参数
        self.confidence_thres = confidence_thres
        self.nms_thres = nms_thres
        self.min_box_hw = min_box_hw  # 改为实例属性
        self.yolo_type = yolo_type

        # 预处理中间变量
        self.ratio = None  # 缩放比例
        self.pad = None  # 填充量 (width_pad, height_pad)

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        """保持长宽比的resize+填充（YOLO标准预处理）"""
        # 重写父类中的resize。就是letterbox
        h, w = image.shape[:2]
        new_h, new_w = self.input_size

        # 计算缩放比例和填充量
        scale = min(new_h / h, new_w / w)
        self.ratio = (scale, scale)

        # 计算新尺寸和填充
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        self.pad = ((new_w - resized_w) // 2, (new_h - resized_h) // 2)

        # 缩放图像并填充
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top=self.pad[1],
            bottom=self.pad[1] + (new_h - resized_h) % 2,  # 处理奇数像素
            left=self.pad[0],
            right=self.pad[0] + (new_w - resized_w) % 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        return padded

    def preprocess_normalize(self, image: np.ndarray) -> np.ndarray:
        # 重写归一化的方法
        return image.astype(np.float32) / 255.0  # 仅归一化到[0,1]

    def _process_box_chunk(self, predictions: np.ndarray, chunk_indices: List[int],
                           ratio: Tuple[float, float], pad: Tuple[float, float]) -> List[Dict[str, Any]]:
        """线程工作函数（修正中心点计算）"""
        chunk_results = []
        ratio_w, ratio_h = ratio
        pad_w, pad_h = pad

        for i in chunk_indices:
            if self.yolo_type == "yolov8":
                # 获取类别得分
                classes_scores = predictions[i, 4:]
                max_score = np.max(classes_scores)
                class_id = np.argmax(classes_scores)  # 记录yolov8的类别编号

                # 置信度过滤
                if max_score < self.confidence_thres:
                    continue
                confidence = max_score
            else:
                # 计算目标置信度
                obj_score = predictions[i, 4]
                classes_scores = predictions[i, 5:]
                max_score = obj_score * np.max(classes_scores)  # 得到最大得分

                # 置信度过滤
                if max_score < self.confidence_thres:
                    continue
                # 记录yolov5的类别编号
                class_id = np.argmax(classes_scores)

            # 计算修正后的框坐标
            cx, cy, w, h = predictions[i, :4]
            x1 = (cx - pad_w) / ratio_w - w / (2 * ratio_w)
            y1 = (cy - pad_h) / ratio_h - h / (2 * ratio_h)
            x2 = x1 + w / ratio_w
            y2 = y1 + h / ratio_h

            # 过滤掉面积太小的框
            if (x2 - x1) * (y2 - y1) < self.min_box_hw[0] * self.min_box_hw[1]:
                continue

            # 保存结果，包括计算后的框和中心点
            chunk_results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': max_score,
                'class_id': class_id,
                'orig_index': i,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]  # 中心点计算
            })

        return chunk_results

    def postprocess_detection(self, outputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """修正后的主流程"""
        # 重写后处理
        # 1. 统一输出格式
        if self.yolo_type == "yolov8":
            predictions = np.squeeze(outputs[0]).T  # (8400, 84)
        else:  # yolov5
            predictions = np.squeeze(outputs[0])  # (25200, 85)

        # 2. 多线程处理
        # 根据线程数，确定每个线程，能够处理的pre box的个数
        # 过滤小框(在原图上进行过滤)
        num_threads = self.num_threads
        chunk_size = predictions.shape[0] // num_threads
        chunks = [list(range(i * chunk_size, (i + 1) * chunk_size if i != num_threads - 1 else predictions.shape[0]))
                  for i in range(num_threads)]

        all_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 分流去调预测逻辑
            futures = [executor.submit(self._process_box_chunk, predictions, chunk, self.ratio, self.pad)
                       for chunk in chunks]
            all_results = [item for future in futures for item in future.result()]

        # 3. 排序和NMS处理
        all_results.sort(key=lambda x: -x['confidence'])

        if not all_results:
            return []

        # 准备NMS输入（修正数组构建方式）
        boxes = np.array([x['bbox'] for x in all_results], dtype=np.float32)
        scores = np.array([x['confidence'] for x in all_results], dtype=np.float32)
        class_ids = np.array([x['class_id'] for x in all_results], dtype=np.int32)
        centers = np.array([x['center'] for x in all_results], dtype=np.float32)  # 中心点数组

        # cv2.dnn.nms()
        # nms有没有遇到过一些问题？
        # 所有类别一起NMS，如果不同类别框1个目标，只会留下1个.
        # 2个框，一个框很大，一个框很小,nms按交并比去不掉.

        # 按类别进行NMS
        # 如果这2个框的中心点距离，在我指定的范围内，就保留面积大的
        # 计算两个框，从相交到最左（最右的距离）,<30px，就去掉小面积的

        keep_indices = self._custom_nms(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            centers=centers,
            iou_threshold=self.nms_thres,
            distance_threshold=self.distance_threshold
        )

        # 4. 组装最终结果
        return [{
            'bbox': [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
            'confidence': float(scores[i]),
            'class_id': int(class_ids[i]),
            'class_name': self.class_names[int(class_ids[i])] if self.class_names else str(int(class_ids[i]))
        } for i in keep_indices]

    def _custom_nms(self, boxes: np.ndarray, scores: np.ndarray,
                    class_ids: np.ndarray, centers: np.ndarray,
                    iou_threshold: float = 0.45,
                    distance_threshold: float = 30.0) -> List[int]:
        """修正索引越界问题的NMS实现"""
        if len(boxes) == 0:
            return []

        # 按置信度降序排序并保留原始索引
        original_indices = np.arange(len(scores))
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        class_ids = class_ids[order]
        centers = centers[order]
        original_indices = original_indices[order]  # 跟踪原始位置
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        keep = []
        while len(order) > 0:
            # 当前最高分框的原始索引
            keep.append(original_indices[0])

            if len(order) == 1:
                break

            # 向量化计算
            xx1 = np.maximum(boxes[0, 0], boxes[1:, 0])
            yy1 = np.maximum(boxes[0, 1], boxes[1:, 1])
            xx2 = np.minimum(boxes[0, 2], boxes[1:, 2])
            yy2 = np.minimum(boxes[0, 3], boxes[1:, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[0] + areas[1:] - intersection + 1e-7)

            # 中心点距离计算
            distances = np.linalg.norm(centers[1:] - centers[0], axis=1)
            suppress_mask = (class_ids[1:] == class_ids[0]) & (
                    (iou > iou_threshold) | (distances < distance_threshold))

            # 更新数组（保留未被抑制的框）
            order = order[1:][~suppress_mask]
            boxes = boxes[1:][~suppress_mask]
            scores = scores[1:][~suppress_mask]
            class_ids = class_ids[1:][~suppress_mask]
            centers = centers[1:][~suppress_mask]
            areas = areas[1:][~suppress_mask]
            original_indices = original_indices[1:][~suppress_mask]

        return keep

    def visualize_detections(
            self,
            image: np.ndarray,
            detections: List[Dict[str, Any]],
            thickness: int = 2,
            font_scale: float = 0.6,
            show_confidence: bool = True,
            show_class: bool = True
    ) -> np.ndarray:
        """
        将检测结果绘制到原图上
        :param image: 原始BGR图像
        :param detections: detect()返回的结果列表
        :param thickness: 框线粗细
        :param font_scale: 字体大小
        :param show_confidence: 是否显示置信度
        :param show_class: 是否显示类别
        :return: 绘制后的图像
        """
        # 创建副本避免修改原图
        vis_img = image.copy()

        # 定义颜色方案（可按类别自定义）
        color_palette = {
            'default': (0, 255, 0),  # 绿色
            'text': (255, 255, 255)  # 白色
        }

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])

            # 绘制边界框
            cv2.rectangle(
                vis_img,
                (x1, y1), (x2, y2),
                color_palette['default'],
                thickness
            )

            # 准备显示文本
            text_parts = []
            if show_class:
                text_parts.append(det.get('class_name', str(det['class_id'])))
            if show_confidence:
                text_parts.append(f"{det['confidence']:.2f}")
            text = " ".join(text_parts)

            # 计算文本位置
            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            # 绘制文本背景
            cv2.rectangle(
                vis_img,
                (x1, y1 - text_h - 5),
                (x1 + text_w, y1),
                color_palette['default'],
                -1  # 填充矩形
            )

            # 绘制文本
            cv2.putText(
                vis_img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color_palette['text'],
                thickness,
                cv2.LINE_AA
            )

        return vis_img


class ImageYoloSegmentation(AIInferenceBase):
    def __init__(self,
                 model_path: str,
                 class_names: Optional[List[str]] = None,
                 use_gpu: bool = False,
                 confidence_thres: float = 0.25,
                 mask_threshold: float = 0.5,
                 nms_thres: float = 0.1,
                 num_threads: int = 5,
                 distance_threshold: int = 30,
                 task_type: str = "yolo-segmentation",
                 min_box_hw: Tuple[int, int] = (10, 10),  # 修改为Tuple类型
                 yolo_type: str = "yolov8"):
        """
        :param min_box_hw: 最小检测框尺寸 (height, width)
        """
        super().__init__(
            model_path=model_path,
            input_size=(640, 640),
            use_gpu=use_gpu,
            class_names=class_names or [],
            task_type=task_type
        )
        self.mask_threshold = mask_threshold
        self.num_threads = num_threads
        self.distance_threshold = distance_threshold
        # 检测参数
        self.confidence_thres = confidence_thres
        self.nms_thres = nms_thres
        self.min_box_hw = min_box_hw  # 改为实例属性
        self.yolo_type = yolo_type

        # 预处理中间变量
        self.ratio = None  # 缩放比例
        self.pad = None  # 填充量 (width_pad, height_pad)

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        """保持长宽比的resize+填充（YOLO标准预处理）"""
        h, w = image.shape[:2]
        new_h, new_w = self.input_size

        # 计算缩放比例和填充量
        scale = min(new_h / h, new_w / w)
        self.ratio = (scale, scale)

        # 计算新尺寸和填充
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        self.pad = ((new_w - resized_w) // 2, (new_h - resized_h) // 2)

        # 缩放图像并填充
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top=self.pad[1],
            bottom=self.pad[1] + (new_h - resized_h) % 2,  # 处理奇数像素
            left=self.pad[0],
            right=self.pad[0] + (new_w - resized_w) % 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        return padded

    def preprocess_normalize(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0  # 仅归一化到[0,1]

    def _custom_nms(self, boxes: np.ndarray, scores: np.ndarray,
                    class_ids: np.ndarray, centers: np.ndarray,
                    iou_threshold: float = 0.45,
                    distance_threshold: float = 30.0) -> List[int]:
        """修正索引越界问题的NMS实现"""
        if len(boxes) == 0:
            return []

        # 按置信度降序排序并保留原始索引
        original_indices = np.arange(len(scores))
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]
        class_ids = class_ids[order]
        centers = centers[order]
        original_indices = original_indices[order]  # 跟踪原始位置
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        keep = []
        while len(order) > 0:
            # 当前最高分框的原始索引
            keep.append(original_indices[0])

            if len(order) == 1:
                break

            # 向量化计算
            xx1 = np.maximum(boxes[0, 0], boxes[1:, 0])
            yy1 = np.maximum(boxes[0, 1], boxes[1:, 1])
            xx2 = np.minimum(boxes[0, 2], boxes[1:, 2])
            yy2 = np.minimum(boxes[0, 3], boxes[1:, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[0] + areas[1:] - intersection + 1e-7)

            # 中心点距离计算
            distances = np.linalg.norm(centers[1:] - centers[0], axis=1)
            suppress_mask = (class_ids[1:] == class_ids[0]) & (
                    (iou > iou_threshold) | (distances < distance_threshold))

            # 更新数组（保留未被抑制的框）
            order = order[1:][~suppress_mask]
            boxes = boxes[1:][~suppress_mask]
            scores = scores[1:][~suppress_mask]
            class_ids = class_ids[1:][~suppress_mask]
            centers = centers[1:][~suppress_mask]
            areas = areas[1:][~suppress_mask]
            original_indices = original_indices[1:][~suppress_mask]

        return keep

    def _process_box_chunk(self, predictions: np.ndarray, chunk_indices: List[int],
                           ratio: Tuple[float, float], pad: Tuple[float, float]) -> List[Dict[str, Any]]:
        """线程工作函数（修正中心点计算）"""
        chunk_results = []
        ratio_w, ratio_h = ratio
        pad_w, pad_h = pad

        for i in chunk_indices:
            # 1. 置信度过滤
            classes_scores = predictions[i, 4:-32]
            max_score = np.max(classes_scores)
            if max_score < self.confidence_thres:
                continue

            # 2. 计算框坐标（修正公式）
            cx, cy, w, h = predictions[i, :4]
            x1 = (cx - pad_w) / ratio_w - w / (2 * ratio_w)
            y1 = (cy - pad_h) / ratio_h - h / (2 * ratio_h)
            x2 = x1 + w / ratio_w
            y2 = y1 + h / ratio_h

            # 3. 尺寸过滤
            if (x2 - x1) * (y2 - y1) < self.min_box_hw[0] * self.min_box_hw[1]:
                continue

            # 4. 保存结果（添加中心点计算）
            chunk_results.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': max_score,
                'class_id': np.argmax(classes_scores),
                'orig_index': i,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            })
        return chunk_results

    def postprocess_yolo_segmentation(self, outputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """YOLOv8分割模型的后处理"""
        # 1. 解包输出
        predictions, mask_protos = outputs[0], outputs[1]
        predictions = np.squeeze(predictions).T  # (8400, 116)
        mask_protos = np.squeeze(mask_protos)  # (32, 160, 160)

        # 2. 多线程处理预测框
        num_threads = self.num_threads
        chunk_size = predictions.shape[0] // num_threads
        chunks = [list(range(i * chunk_size, (i + 1) * chunk_size if i != num_threads - 1 else predictions.shape[0]))
                  for i in range(num_threads)]

        all_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._process_box_chunk, predictions, chunk, self.ratio, self.pad)
                       for chunk in chunks]
            all_results = [item for future in futures for item in future.result()]

        # 3. 排序和NMS处理
        all_results.sort(key=lambda x: -x['confidence'])

        if not all_results:
            return []

        # 准备NMS输入
        boxes = np.array([x['bbox'] for x in all_results], dtype=np.float32)
        scores = np.array([x['confidence'] for x in all_results], dtype=np.float32)
        class_ids = np.array([x['class_id'] for x in all_results], dtype=np.int32)
        centers = np.array([x['center'] for x in all_results], dtype=np.float32)

        keep_indices = self._custom_nms(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            centers=centers,
            iou_threshold=self.nms_thres,
            distance_threshold=self.distance_threshold
        )

        # 4. 处理分割mask
        final_results = []
        input_h, input_w = self.input_size  # 模型输入尺寸
        mask_dim = mask_protos.shape[0]  # mask原型数量(32)

        for idx in keep_indices:
            result = all_results[idx]
            box = result['bbox']
            # box坐标已经是原图(1290x1080)坐标系
            x1, y1, x2, y2 = box

            # 获取mask系数并计算原始mask(160x160)  8229 160x160x32
            mask_coeff = predictions[result['orig_index'], -mask_dim:]

            mask = (mask_protos.transpose(1, 2, 0) @ mask_coeff).squeeze()
            mask = 1 / (1 + np.exp(-mask))  # sigmoid激活

            # 关键步骤：将原图box坐标转换到160x160 mask坐标系
            # 1. 原图 -> 输入尺寸(640x640)
            scaled_x1 = (x1 * self.ratio[0]) + self.pad[0]
            scaled_y1 = (y1 * self.ratio[1]) + self.pad[1]
            scaled_x2 = (x2 * self.ratio[0]) + self.pad[0]
            scaled_y2 = (y2 * self.ratio[1]) + self.pad[1]

            # 2. 640x640 -> 160x160 (mask尺寸)
            mask_scale = 160 / self.input_size[0]  # mask下采样比例
            mask_x1 = int(scaled_x1 * mask_scale)
            mask_y1 = int(scaled_y1 * mask_scale)
            mask_x2 = int(scaled_x2 * mask_scale)
            mask_y2 = int(scaled_y2 * mask_scale)

            # 确保不越界
            mask_x1 = max(0, mask_x1)
            mask_y1 = max(0, mask_y1)
            mask_x2 = min(160, mask_x2)
            mask_y2 = min(160, mask_y2)
            mask_w = mask_x2 - mask_x1
            mask_h = mask_y2 - mask_y1

            if mask_w > 0 and mask_h > 0:
                # 从160x160 mask中提取对应区域 54:132, 55:95
                roi_mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

                # 计算原图上实际的box尺寸
                box_w = x2 - x1
                box_h = y2 - y1

                # 将mask缩放到原图box的实际大小
                roi_mask = cv2.resize(roi_mask, (int(box_w), int(box_h)),
                                      interpolation=cv2.INTER_LINEAR)
                roi_mask = cv2.GaussianBlur(roi_mask, (5, 5), 0)
                kernel = np.ones((5, 5), np.uint8)
                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
                roi_mask = (roi_mask > self.mask_threshold).astype(np.uint8) * 255
                # show(roi_mask,'roi_mask')
                # 存储结果
                result['mask'] = {
                    'roi_mask': roi_mask,
                    'box_coords': (int(x1), int(y1), int(x2), int(y2))  # 原图坐标
                }
            else:
                result['mask'] = None

            final_results.append(result)

        return final_results

    def visualize_detections(
            self,
            image: np.ndarray,
            detections: List[Dict[str, Any]],
            thickness: int = 2,
            font_scale: float = 0.6,
            show_confidence: bool = True,
            show_class: bool = True,
            show_mask: bool = True,
            mask_alpha: float = 0.3,
            random_color_mask: bool = True,
            draw_contour: bool = True
    ) -> np.ndarray:
        vis_img = image.copy()
        h, w = vis_img.shape[:2]

        # 高对比度颜色
        color_palette = [
            (0, 255, 255), (255, 0, 255), (0, 165, 255),
            (255, 0, 0), (0, 255, 0)
        ]
        rng = np.random.default_rng() if random_color_mask else None

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = tuple(map(int, rng.integers(0, 256, size=3))) if random_color_mask else color_palette[
                i % len(color_palette)]

            # 1. 处理mask
            if show_mask and 'mask' in det and det['mask'] is not None:
                mask_data = det['mask']
                roi_mask = mask_data['roi_mask']
                # if True:
                #     cv2.imshow('',roi_mask)
                #     cv2.waitKey(0)
                #     cannyImg = cv2.Canny(roi_mask, 125, 200)
                #     cv2.imshow('canny2', cannyImg)
                #     cv2.waitKey(0)
                box_x1, box_y1, box_x2, box_y2 = mask_data['box_coords']

                # 计算实际可用区域
                roi_h, roi_w = roi_mask.shape
                target_h = min(roi_h, box_y2 - box_y1)
                target_w = min(roi_w, box_x2 - box_x1)

                if draw_contour:
                    # 轮廓模式
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    temp_mask[box_y1:box_y1 + target_h, box_x1:box_x1 + target_w] = roi_mask[:target_h, :target_w]
                    contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 1004,3233
                    cv2.drawContours(vis_img, contours, -1, color, thickness)
                else:
                    # 填充模式
                    roi = vis_img[box_y1:box_y1 + target_h, box_x1:box_x1 + target_w]
                    mask_area = roi_mask[:target_h, :target_w] > 0
                    vis_img[box_y1:box_y1 + target_h, box_x1:box_x1 + target_w][mask_area] = (
                            roi[mask_area] * (1 - mask_alpha) + np.array(color) * mask_alpha
                    ).astype(np.uint8)

            # 2. 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)

            # 3. 绘制文本
            text_parts = []
            if show_class:
                index = det['class_id']
                text_parts.append(self.class_names[index]+f"_{index}")
                # text_parts.append(det.get('class_name', f"Class_{det['class_id']}"))
            if show_confidence:
                text_parts.append(f"{det['confidence']:.2f}")
            text = " ".join(text_parts)

            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_bg_y1 = max(0, y1 - text_h - 5)

            cv2.rectangle(
                vis_img, (x1, text_bg_y1), (x1 + text_w, y1),
                (255, 255, 255), -1
            )
            cv2.putText(
                vis_img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness, cv2.LINE_AA
            )

        return vis_img


class ImageMaskSegmentation(AIInferenceBase):
    def __init__(self,
                 model_path: str,
                 class_names: Optional[List[str]] = None,
                 use_gpu: bool = False,
                 distance_threshold: int = 30,
                 task_type: str = "mask-segmentation",
                 nms_cls: str = "all",
                 nms_ths: float = 0.2,
                 min_box_hw: Tuple[int, int] = (10, 10)):
        super().__init__(
            model_path=model_path,
            input_size=(512, 512),
            use_gpu=use_gpu,
            class_names=class_names or [],
            task_type=task_type
        )
        self.nms_ths = nms_ths
        self.min_box_hw = min_box_hw
        self.scale = None
        self.padw = None
        self.padh = None
        self.orgW, self.orgH = None, None
        self.distance_threshold = distance_threshold
        assert len(self.class_names) > 0, "必须输入入class_names"
        self.nms_cls = nms_cls

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        iw, ih = image.shape[1], image.shape[0]
        self.orgW = iw
        self.orgH = ih

        w, h = self.input_size
        self.scale = min(w / iw, h / ih)
        nw = int(iw * self.scale)
        nh = int(ih * self.scale)
        resized_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        x_offset = (w - nw) // 2
        y_offset = (h - nh) // 2
        self.padw = nw
        self.padh = nh
        new_image[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized_image
        return new_image

    def preprocess_normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化图像"""
        image = image.astype(np.float32)
        image -= np.array([123.675, 116.28, 103.53], np.float32)
        image /= np.array([58.395, 57.12, 57.375], np.float32)
        return image

    def _crop_center(self, softmax_max):
        # 只保留中间部分，去喂入网络时的128的区域
        start_h = (self.input_size[0] - self.padh) // 2
        start_w = (self.input_size[1] - self.padw) // 2
        cropped_pr = softmax_max[start_h:start_h + self.padh, start_w:start_w + self.padw]
        return cropped_pr

    def softmax(self, x, axis=-1):
        # 计算 softmax，防止溢出，先减去最大值
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def postprocess_mask_segmentation(self, outputs: List[np.ndarray]) -> np.ndarray:
        """分割后处理（统一接口）"""
        outputs = np.squeeze(outputs[0], axis=0)  # 14x512x512
        outputs = np.transpose(outputs, [1, 2, 0])
        softmax_max = self.softmax(outputs, axis=-1)
        cropped_out = self._crop_center(softmax_max)
        orgin_pre = cv2.resize(cropped_out, (self.orgW, self.orgH), cv2.INTER_LINEAR)
        pre_class_index = orgin_pre.argmax(axis=-1)
        class_masks = {}
        boxes = []
        # 第一阶段：生成原始boxes和masks（保持不变）
        for i in range(len(self.class_names)):
            if i == 0: continue
            binary_mask = (pre_class_index == i).astype(np.uint8)
            roi_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

            x, y, w, h = cv2.boundingRect(binary_mask)
            if w * h < self.min_box_hw[0] * self.min_box_hw[1]:
                continue
            boxes.append([i, x, y, w, h])
            class_masks[i] = binary_mask

        # 第二阶段：按模式进行NMS过滤
        mask_result = []
        if len(boxes) > 0:
            boxes_array = np.array(boxes)
            if self.nms_cls == "class":  # 模式1：仅同类过滤
                unique_classes = np.unique(boxes_array[:, 0])
                filtered_boxes = []
                for cls in unique_classes:
                    cls_boxes = boxes_array[boxes_array[:, 0] == cls]
                    filtered = self._apply_nms(cls_boxes)
                    filtered_boxes.extend(filtered.tolist())
                boxes = filtered_boxes

            elif self.nms_cls == "all":  # 模式2：全局过滤
                boxes = self._apply_nms_cross_class(boxes_array).tolist()

            # 生成最终mask_result
            for box in boxes:
                mask_result.append(class_masks[box[0]])

        return (boxes, mask_result)

    def _apply_nms_cross_class(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.astype(float)
        n = len(boxes)
        keep = np.ones(n, dtype=bool)

        x1, y1 = boxes[:, 1], boxes[:, 2]
        x2, y2 = x1 + boxes[:, 3], y1 + boxes[:, 4]
        areas = boxes[:, 3] * boxes[:, 4]

        for i in range(n):
            if not keep[i]: continue

            for j in range(i + 1, n):
                if not keep[j]: continue

                # 计算IoU
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[j] - inter)

                # 场景1：检查包含关系（允许少量偏移）
                offset_threshold = self.distance_threshold
                i_contains_j = (x1[i] <= x1[j] + offset_threshold) and \
                               (y1[i] <= y1[j] + offset_threshold) and \
                               (x2[i] >= x2[j] - offset_threshold) and \
                               (y2[i] >= y2[j] - offset_threshold)
                j_contains_i = (x1[j] <= x1[i] + offset_threshold) and \
                               (y1[j] <= y1[i] + offset_threshold) and \
                               (x2[j] >= x2[i] - offset_threshold) and \
                               (y2[j] >= y2[i] - offset_threshold)

                if i_contains_j or j_contains_i:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    continue

                # 新增场景1.5：小框左上角靠近大框中心（距离<30像素）
                center_i = np.array([(x1[i] + x2[i]) / 2, (y1[i] + y2[i]) / 2])
                center_j = np.array([(x1[j] + x2[j]) / 2, (y1[j] + y2[j]) / 2])
                small_box, large_box = (i, j) if areas[i] < areas[j] else (j, i)

                # 计算小框左上角到大框中心的距离
                small_box_tl = np.array([x1[small_box], y1[small_box]])  # 左上角 (x1,y1)
                large_box_center = center_i if large_box == i else center_j
                dist_tl_to_center = np.linalg.norm(small_box_tl - large_box_center)

                ratio = dist_tl_to_center // self.distance_threshold

                if dist_tl_to_center < self.distance_threshold * (ratio // 2):  # 阈值设为30像素
                    keep[small_box] = False
                    if small_box == i:  # 如果当前i是小框，直接跳出j循环
                        break
                    continue

                # 场景2：检查IoU（跨类别）
                if iou > self.nms_ths:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    continue

                # 场景3：检查中心点距离（跨类别）
                dist_center_to_center = np.linalg.norm(center_i - center_j)
                if dist_center_to_center < self.distance_threshold:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return boxes[keep].astype(int)

    def _apply_nms(self, boxes: np.ndarray) -> np.ndarray:
        """增强版NMS核心方法，处理三种场景：
        1. 中心点距离过近的框（基于distance_threshold）
        2. 完全被包含的框（无论距离）
        3. 重叠度过高的框（基于iou_threshold）
        """
        boxes = boxes.astype(float)
        n = len(boxes)
        keep = np.ones(n, dtype=bool)

        # 预计算框参数 [x1,y1,x2,y2]格式
        x1, y1 = boxes[:, 1], boxes[:, 2]
        x2, y2 = x1 + boxes[:, 3], y1 + boxes[:, 4]
        areas = boxes[:, 3] * boxes[:, 4]

        for i in range(n):
            if not keep[i]: continue

            for j in range(i + 1, n):
                if not keep[j]: continue

                # 计算IoU
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[j] - inter)

                # 场景1：检查包含关系（最高优先级）
                i_contains_j = (x1[i] <= x1[j]) and (y1[i] <= y1[j]) and \
                               (x2[i] >= x2[j]) and (y2[i] >= y2[j])
                j_contains_i = (x1[j] <= x1[i]) and (y1[j] <= y1[i]) and \
                               (x2[j] >= x2[i]) and (y2[j] >= y2[i])

                if i_contains_j or j_contains_i:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    continue

                # 场景2：检查IoU（中等优先级）
                if iou > self.nms_ths:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    continue

                # 场景3：检查中心点距离（最低优先级）
                center_i = [(x1[i] + x2[i]) / 2, (y1[i] + y2[i]) / 2]
                center_j = [(x1[j] + x2[j]) / 2, (y1[j] + y2[j]) / 2]
                dist = np.sqrt((center_i[0] - center_j[0]) ** 2 +
                               (center_i[1] - center_j[1]) ** 2)

                if dist < self.distance_threshold:
                    if areas[i] >= areas[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return boxes[keep].astype(int)

    def visualize_detections(
            self,
            image: np.ndarray,
            boxes: List[List[int]],
            mask_result: List[np.ndarray],
            thickness: int = 2,
            font_scale: float = 0.6,
            show_confidence: bool = False,  # 兼容无置信度的情况
            show_class: bool = True,
            show_mask: bool = True,
            mask_alpha: float = 0.3,
            random_color: bool = False
    ) -> np.ndarray:
        """
        可视化检测结果和分割mask
        Args:
            image: 原始BGR图像 (H,W,3)
            boxes: [[class_id,x,y,w,h],...]
            mask_result: 对应的mask列表 [H,W]uint8
            thickness: 框线粗细
            font_scale: 字体大小
            show_class: 是否显示类别
            show_mask: 是否显示分割mask
            mask_alpha: mask透明度(0-1)
            random_color: 是否为每个实例随机颜色
        Returns:
            可视化后的BGR图像
        """
        vis_img = image.copy()
        h, w = vis_img.shape[:2]

        # 颜色方案 (COCO标准色+随机色选项)
        color_palette = [
            (241, 23, 78),  # 红
            (0, 255, 255),  # 黄
            (0, 206, 209),  # 青
            (255, 0, 255),  # 粉
            (0, 255, 0)  # 绿
        ]
        rng = np.random.default_rng() if random_color else None

        for i, (box, mask) in enumerate(zip(boxes, mask_result)):
            class_id, x, y, box_w, box_h = map(int, box)

            # 生成颜色
            if random_color:
                color = tuple(map(int, rng.integers(0, 256, size=3)))
            else:
                color = color_palette[class_id % len(color_palette)]

            # 绘制mask
            if show_mask and mask is not None:
                # 确保mask尺寸匹配
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # 创建彩色mask
                colored_mask = np.zeros_like(vis_img)
                colored_mask[mask > 0] = color

                # 与原图混合
                vis_img = cv2.addWeighted(colored_mask, mask_alpha, vis_img, 1 - mask_alpha, 0)

            # 绘制边界框
            cv2.rectangle(vis_img, (x, y), (x + box_w, y + box_h), color, thickness)

            # 绘制类别标签
            if show_class:
                label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                text = f"{label}"

                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # 文本背景
                cv2.rectangle(
                    vis_img,
                    (x, y - text_h - 5),
                    (x + text_w, y),
                    color, -1
                )
                # 文本
                cv2.putText(
                    vis_img, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )

        return vis_img





if __name__ == "__main__":
    path = r"D:\testcolor\error10.jpg"
    # path = r"F:\data_source\Human_Parsing_Dataset-20250608T082454Z-1-001\closes\Deepfashion2\train\JPEGImages/000002.jpg"
    # path = r'C:\Users\qwen\Desktop\11\3.jpg'
    model_path = r"F:\2025\py_do2025\AnalyzeColor\clothes_class.onnx"
    yolov8_path = r"F:\data_source\diandongche\ultralytics11\yolo11s.onnx"
    yolov8_seg_path = r"F:\data_source\diandongche\ultralytics11\yolo11n-seg.onnx"
    mask_seg_path = r"F:\2025\py_do2025\ClosesSeg\segformer-pytorch-master\model_data\ep080-loss.onnx"
    rtmo_path = r"F:\2025\py_do2025\AnalyzeColor\Point3DPerson\rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
    img = cv2.imread(path, 1)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cannyImg = cv2.Canny(gray,125,200)
    # cv2.imshow('canny',cannyImg)
    # cv2.waitKey(0)
    # img_cls = ImageClassifier(model_path)
    # result = img_cls.predict(img)
    # print(result)

    yolo_model = ImageYoloDetect(yolov8_path)
    result = yolo_model.predict(img)

    vis_img = yolo_model.visualize_detections(img, result, show_class=False)
    vis_img = cv2.resize(vis_img, [640, 640])
    cv2.imshow('vis_img',vis_img)
    cv2.waitKey(0)

    #
    # point_model = ImageRTMOKeyPoints(rtmo_path,kpt_thr=2)
    # keypoints, scores = [], []
    # # 每个box进行处理
    # for box in result:
    #     xyxy_bbox = box['bbox']
    #     keypoint, score = point_model.predict(xyxy_bbox, img)
    #     # 得到关键点的外接bbox坐标
    #     kpts = keypoint[0]
    #     bbox = pose_to_bbox(kpts)
    #     img = point_model.visualize_detections(img,keypoint, score)
    # result_img = cv2.resize(img, [640, 640])
    # cv2.imshow('vis_img',result_img)
    # cv2.waitKey(0)
    pass

    # vis_img = yolo_model.visualize_detections(img, result, show_class=False)
    # vis_img = cv2.resize(vis_img, [640, 640])
    # cv2.imshow('vis_img',vis_img)
    # cv2.waitKey(0)

    # yolo_seg_model = ImageYoloSegmentation(yolov8_seg_path)
    # detections = yolo_seg_model.predict(img)
    # result_img = yolo_seg_model.visualize_detections(
    #     img,
    #     detections,
    #     show_mask=True,
    #     mask_alpha=0.5,
    #     draw_contour=True
    # )
    # result_img = cv2.resize(result_img, [640, 640])
    # cv2.imshow('vis_img',result_img)
    # cv2.waitKey(0)
    #
    # class_names = [
    #     "Background",  # 0
    #     "Short-sleeve top",  # 1
    #     "Long-sleeve top",  # 2
    #     "Short-sleeve jacket",  # 3
    #     "Long-sleeve jacket",  # 4
    #     "Vest",  # 5
    #     "Spaghetti strap",  # 6
    #     "Shorts",  # 7
    #     "Long pants",  # 8
    #     "Skirt",  # 9
    #     "Short-sleeve dress",  # 10
    #     "Long-sleeve dress",  # 11
    #     "Vest dress",  # 12
    #     "Spaghetti strap dress",  # 13
    # ]
    # mask_seg_model = ImageMaskSegmentation(
    #     mask_seg_path,
    #     class_names=class_names,
    #     nms_cls='all',
    #     nms_ths=0.1,
    #     distance_threshold=30,
    #     min_box_hw=[10, 10]
    # )
    # #
    # boxes, mask_result = mask_seg_model.predict(img)
    # print(boxes)
    # result_img = mask_seg_model.visualize_detections(
    #     img,
    #     boxes,
    #     mask_result,
    #     show_mask=True,
    #     mask_alpha=0.4,
    #     random_color=True
    # )
    # result_img = cv2.resize(result_img, [640, 640])
    # cv2.imshow('vis_img', result_img)
    # cv2.waitKey(0)
    pass
