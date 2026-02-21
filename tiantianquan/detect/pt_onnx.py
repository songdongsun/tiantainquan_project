from tiantianquan.detect.ai_inferenceBase import *


def process_segmentation_request(img, seg_request, sub_inner_radio=100, add_outter_raio=20):
    """
    处理分割请求，对置信度最高的class_id=1对象绘制内接圆并填充，
    对置信度最高的class_id=0对象进行外接矩形扩展并裁剪

    参数:
    img: 原始图像 (numpy数组)
    seg_request: 分割结果列表，每个元素包含class_id、confidence和mask信息

    返回:
    processed_img: 处理后的图像
    cropped_img: 裁剪出的图像 (如果没有class_id=0的对象则返回None)
    crop_coords: 裁剪坐标 (x1, y1, x2, y2) 或 None
    """
    # 复制原始图像，避免修改原图
    processed_img = img.copy()

    # 分离class_id=0和1的对象，并找出每个类别中置信度最高的对象
    class0_objs = [obj for obj in seg_request if obj['class_id'] == 0]
    class1_objs = [obj for obj in seg_request if obj['class_id'] == 1]

    # 找出置信度最高的class_id=1对象并绘制内接圆
    if class1_objs:
        best_class1 = max(class1_objs, key=lambda x: x['confidence'])
        mask_data = best_class1['mask']
        roi_mask = mask_data['roi_mask']
        box_x1, box_y1, box_x2, box_y2 = mask_data['box_coords']

        # 确保掩码是uint8类型，值为0和255
        if roi_mask.dtype != np.uint8:
            roi_mask = roi_mask.astype(np.uint8)

        # 计算实际可用区域
        roi_h, roi_w = roi_mask.shape
        target_h = min(roi_h, box_y2 - box_y1)
        target_w = min(roi_w, box_x2 - box_x1)

        # 创建一个临时全图掩码
        temp_mask = np.zeros((processed_img.shape[0], processed_img.shape[1]), dtype=np.uint8)
        temp_mask[box_y1:box_y1 + target_h, box_x1:box_x1 + target_w] = roi_mask[:target_h, :target_w]

        # 计算最小外接圆
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找到最大轮廓
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)

            # 绘制内接圆（半径减少50像素）并填充为114
            new_radius = max(0, int(radius) - sub_inner_radio)  # 确保半径不小于0
            if new_radius > 0:
                # 确定填充颜色
                if len(processed_img.shape) == 3:  # 彩色图像
                    color = (114, 114, 114)
                else:  # 灰度图像
                    color = 114

                # 直接在图像上绘制填充圆
                cv2.circle(processed_img, (int(x), int(y)), new_radius, color, -1)

    # 找出置信度最高的class_id=0对象并裁剪
    cropped_img = None
    crop_coords = None
    if class0_objs:
        best_class0 = max(class0_objs, key=lambda x: x['confidence'])
        mask_data = best_class0['mask']
        roi_mask = mask_data['roi_mask']
        box_x1, box_y1, box_x2, box_y2 = mask_data['box_coords']

        # 确保掩码是uint8类型，值为0和255
        if roi_mask.dtype != np.uint8:
            roi_mask = roi_mask.astype(np.uint8)

        # 计算实际可用区域
        roi_h, roi_w = roi_mask.shape
        target_h = min(roi_h, box_y2 - box_y1)
        target_w = min(roi_w, box_x2 - box_x1)

        # 创建一个临时全图掩码
        temp_mask = np.zeros((processed_img.shape[0], processed_img.shape[1]), dtype=np.uint8)
        temp_mask[box_y1:box_y1 + target_h, box_x1:box_x1 + target_w] = roi_mask[:target_h, :target_w]

        # 使用cv2.findNonZero找到非零点的坐标
        points = cv2.findNonZero(temp_mask)
        if points is not None:
            # 获取外接矩形
            x, y, w, h = cv2.boundingRect(points)

            # 扩展矩形50像素
            x1 = max(0, x - add_outter_raio)
            y1 = max(0, y - add_outter_raio)
            x2 = min(processed_img.shape[1], x + w + add_outter_raio)  # 图像宽度
            y2 = min(processed_img.shape[0], y + h + add_outter_raio)  # 图像高度

            # 记录裁剪坐标
            crop_coords = (x1, y1, x2, y2)

            # 裁剪图像
            if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                cropped_img = processed_img[y1:y2, x1:x2]

    return processed_img, cropped_img, crop_coords


def sliding_crop(image, num=(2, 2), overlap=250):
    """
    滑动裁图,裁成num[0]*num[1]个图，不处理标签框

    :param image: 输入图像
    :param num: 竖向滑动num[0]格，横向滑动num[1]格
    :param overlap: 重叠多少像素
    :return: 字典，键为裁剪图像索引，值为包含裁剪图像和位置坐标的元组 (crop_image, (x, y, x2, y2))
    """
    img_height, img_width = image.shape[:2]

    # 计算裁剪区域大小和步长
    crop_height, crop_width = int(img_height / num[0] + overlap / 2), int(img_width / num[1] + overlap / 2)
    step_h, step_w = crop_height - overlap, crop_width - overlap  # 竖向,横向滑动步长

    crop_results = {}  # 存放裁下来的图和位置信息
    image = image.copy()

    idx = 0
    for y in range(0, img_height - crop_height + 1, step_h):  # 竖向滑动
        for x in range(0, img_width - crop_width + 1, step_w):  # 横向滑动
            # 裁剪图像
            crop_image = image[y:y + crop_height, x:x + crop_width]

            # 记录裁剪位置 (x, y, x2, y2)
            crop_position = (x, y, x + crop_width, y + crop_height)

            # 添加到结果字典
            crop_results[idx] = (crop_image, crop_position)
            idx += 1

    return crop_results


class NGPipeline:
    def __init__(self, seg_model_path, det_model_path, is_debug=True):
        """
        初始化推理类
        """
        self.is_debug = is_debug
        self.seg_model = ImageYoloSegmentation(seg_model_path, class_names={0: "outter", 1: "inner"})
        self.det_model = ImageYoloDetect(det_model_path, class_names={0: "pt_ng"},
                                         yolo_type="yolov5", model_size=(1120, 1120))

    def segment_image(self, image):
        """
        执行分割，返回分割结果和可视化图像
        """
        seg_result = self.seg_model.predict(image)
        if self.is_debug:
            result_img = self.seg_model.visualize_detections(
                image, seg_result,
                show_mask=True, mask_alpha=0.5, draw_contour=True
            )
            cv2.imwrite("./result.jpg", result_img)
        return seg_result

    def crop_image(self, image, seg_result):
        """
        根据分割结果裁剪图像
        """
        processed_img, cropped_img, crop_coords = process_segmentation_request(image, seg_result)
        crop_results = sliding_crop(cropped_img, num=(2, 2), overlap=250)
        return processed_img, cropped_img, crop_coords, crop_results

    def detect_objects(self, image, crop_results, crop_coords):
        """
        在裁剪后的图上检测，并还原到原图坐标
        """
        all_boxes, all_scores, all_class_ids, all_centers = [], [], [], []

        for key, value in crop_results.items():
            detect_img, old_xyxy = value
            if self.is_debug:
                cv2.imwrite(f"{key}.jpg", detect_img)

            ng_result = self.det_model.predict(detect_img)
            if not ng_result:
                continue

            # 坐标映射回原图
            for detection in ng_result:
                bbox = detection['bbox']
                x1_crop = bbox[0] + old_xyxy[0]
                y1_crop = bbox[1] + old_xyxy[1]
                x2_crop = bbox[2] + old_xyxy[0]
                y2_crop = bbox[3] + old_xyxy[1]

                # 映射到原始图像
                x1_orig = x1_crop + crop_coords[0]
                y1_orig = y1_crop + crop_coords[1]
                x2_orig = x2_crop + crop_coords[0]
                y2_orig = y2_crop + crop_coords[1]

                center_x = (x1_orig + x2_orig) / 2
                center_y = (y1_orig + y2_orig) / 2

                all_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
                all_scores.append(detection['confidence'])
                all_class_ids.append(detection['class_id'])
                all_centers.append([center_x, center_y])

        return np.array(all_boxes), np.array(all_scores), np.array(all_class_ids), np.array(all_centers)

    def apply_nms(self, boxes, scores, class_ids, centers):
        """
        对检测结果进行NMS去重
        """
        keep_indices = self.det_model._custom_nms(boxes, scores, class_ids, centers)
        final_results = []
        for idx in keep_indices:
            final_results.append({
                'bbox': boxes[idx].tolist(),
                'confidence': float(scores[idx]),
                'class_id': int(class_ids[idx]),
                'class_name': "pt_ng"
            })
        return final_results

    def visualize(self, image, final_results, save_path="./final_result.jpg"):
        """
        可视化检测结果
        """
        vis_img = image.copy()
        for result in final_results:
            x1, y1, x2, y2 = map(int, result['bbox'])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result['class_name']}: {result['confidence']:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(save_path, vis_img)

    def run(self, image):
        """
        主流程：分割 → 裁剪 → 检测 → NMS → 可视化
        """
        seg_result = self.segment_image(image)
        _, _, crop_coords, crop_results = self.crop_image(image, seg_result)
        boxes, scores, class_ids, centers = self.detect_objects(image, crop_results, crop_coords)

        if len(boxes) == 0:
            print("未检测到结果")
            return []

        final_results = self.apply_nms(boxes, scores, class_ids, centers)

        if self.is_debug:
            self.visualize(image, final_results)

        return final_results


class SegPipeline:
    def __init__(self, seg_model_path, is_debug=True):
        """
        初始化推理类
        """
        self.is_debug = is_debug
        self.seg_model = ImageYoloSegmentation(seg_model_path, class_names={0: "outter", 1: "inner"})


    def segment_image(self, image):
        """
        执行分割，返回分割结果和可视化图像
        """
        seg_result = self.seg_model.predict(image)
        if self.is_debug:
            result_img = self.seg_model.visualize_detections(
                image, seg_result,
                show_mask=True, mask_alpha=0.5, draw_contour=True
            )
            cv2.imwrite("./tiantianquan/pt/segment/results/result2.jpg", result_img)
        return seg_result

    def crop_image(self, image, seg_result):
        """
        根据分割结果裁剪图像
        """
        processed_img, cropped_img, crop_coords = process_segmentation_request(image, seg_result)
        crop_results = sliding_crop(cropped_img, num=(2, 2), overlap=250)

        if self.is_debug:
            cv2.imwrite("./tiantianquan/pt/segment/results/crop_results2.jpg", cropped_img)

        return processed_img, cropped_img, crop_coords, crop_results

    def run(self, image):
        """
        主流程：分割 → 裁剪 → 检测 → NMS → 可视化
        """
        seg_result = self.segment_image(image)
        _, _, crop_coords, crop_results = self.crop_image(image, seg_result)


        return crop_results


if __name__ == "__main__":
    # path = r"test_pt.jpg"
    path = r"test2.jpg"
    image = cv2.imread(path)

    pipeline = NGPipeline(seg_model_path="./models/quan.onnx",
                          det_model_path="./models/pt.onnx",
                          is_debug=False)

    results = pipeline.run(image)
    print("最终检测结果:", results)


#
# if __name__ == "__main__":
#     path = r"test_pt.jpg"
#     quan_seg_model = "./models/quan.onnx"
#     det_model = "./models/pt.onnx"
#     is_debug = True
#     image = cv2.imread(path)
#     quan_seg = ImageYoloSegmentation(quan_seg_model, class_names={0: "outter", 1: "inner"})
#     det_box = ImageYoloDetect(det_model, class_names={0: "pt_ng"}, yolo_type="yolov5", model_size=(1120, 1120))
#     # 在原图中分割得到目标区域，坐标已还原到image中
#     quan_seg_result = quan_seg.predict(image)
#
#     if is_debug:
#         result_img = quan_seg.visualize_detections(
#             image,
#             quan_seg_result,
#             show_mask=True,
#             mask_alpha=0.5,
#             draw_contour=True
#         )
#         # result_img = cv2.resize(result_img, [640, 640])
#         # cv2.imshow('vis_img',result_img)
#         # cv2.waitKey(0)
#         cv2.imwrite("./result.jpg", result_img)
#     # 根据分割得到有效区域图，即cropped_img。crop_coords为记录image中的坐标
#     processed_img, cropped_img, crop_coords = process_segmentation_request(image, quan_seg_result)
#     # 进行滑动平均裁图,crop_results为裁后的图，以及4张图坐标
#     crop_results = sliding_crop(cropped_img, num=(2, 2), overlap=250)
#     detect_result = {}
#     all_boxes = []  # 存储所有检测框（在原始图像中的坐标）
#     all_scores = []  # 存储所有置信度
#     all_class_ids = []  # 存储所有类别ID
#     all_centers = []  # 存储所有中心点
#
#     for key, value in crop_results.items():
#         detect_img, old_xyxy = value
#         if is_debug:
#             cv2.imwrite(f'{key}.jpg', detect_img)
#
#         # 4张图片中如果1张图有目标，则记录结果
#         ng_result = det_box.predict(detect_img)
#         if len(ng_result):
#             detect_result[key] = ng_result
#
#             # 将检测结果从裁剪图坐标转换到原始图像坐标
#             for detection in ng_result:
#                 # 1. 先将坐标从裁剪图转换到cropped_img
#                 bbox = detection['bbox']
#                 x1_crop = bbox[0] + old_xyxy[0]
#                 y1_crop = bbox[1] + old_xyxy[1]
#                 x2_crop = bbox[2] + old_xyxy[0]
#                 y2_crop = bbox[3] + old_xyxy[1]
#
#                 # 2. 再将坐标从cropped_img转换到原始图像
#                 x1_orig = x1_crop + crop_coords[0]
#                 y1_orig = y1_crop + crop_coords[1]
#                 x2_orig = x2_crop + crop_coords[0]
#                 y2_orig = y2_crop + crop_coords[1]
#
#                 # 计算中心点
#                 center_x = (x1_orig + x2_orig) / 2
#                 center_y = (y1_orig + y2_orig) / 2
#
#                 # 存储转换后的结果
#                 all_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
#                 all_scores.append(detection['confidence'])
#                 all_class_ids.append(detection['class_id'])
#                 all_centers.append([center_x, center_y])
#
#     # 如果有检测结果，进行NMS去重
#     if all_boxes:
#         # 转换为numpy数组
#         boxes_np = np.array(all_boxes)
#         scores_np = np.array(all_scores)
#         class_ids_np = np.array(all_class_ids)
#         centers_np = np.array(all_centers)
#
#         # 应用NMS
#         keep_indices = det_box._custom_nms(boxes_np, scores_np, class_ids_np, centers_np)
#
#         # 构建最终结果
#         final_results = []
#         for idx in keep_indices:
#             final_results.append({
#                 'bbox': boxes_np[idx].tolist(),
#                 'confidence': float(scores_np[idx]),
#                 'class_id': int(class_ids_np[idx]),
#                 'class_name': 'pt_ng'  # 根据您的类别映射设置
#             })
#
#         # 打印或使用最终结果
#         print(f"最终检测结果: {final_results}")
#
#         # 如果需要可视化最终结果
#         if is_debug:
#             final_img = image.copy()
#             for result in final_results:
#                 bbox = result['bbox']
#                 cv2.rectangle(final_img,
#                               (int(bbox[0]), int(bbox[1])),
#                               (int(bbox[2]), int(bbox[3])),
#                               (0, 255, 0), 2)
#                 label = f"{result['class_name']}: {result['confidence']:.2f}"
#                 cv2.putText(final_img, label,
#                             (int(bbox[0]), int(bbox[1]) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#             cv2.imwrite("./final_result.jpg", final_img)
#
#     pass
