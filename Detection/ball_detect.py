# non_maximum_suppression(), detect_balls()
import numpy as np
from ultralytics import YOLO

# 중복되는 bbox 제거
def non_maximum_suppression(bboxes, confidences, iou_threshold):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()[::-1]  # Confidence 기준 내림차순 정렬

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # IoU 계산
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / union_area

        # IoU threshold보다 작은 박스만 유지
        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return keep

# 당구공 detect
def detect_balls(frame, corners, iou_threshold=0.6, confidence_threshold=0.4):
    # YOLO 모델 로드
	model = YOLO('real.engine')  # 적절한 YOLO 모델 경로

	# YOLO 객체 탐지
	results = model.predict(source=frame, imgsz=1024)
	detections = results[0].boxes.data.cpu().numpy()  # 감지 결과를 NumPy로 변환

	# 각 바운딩 박스 정보 (x1, y1, x2, y2, confidence, class_id)
	bboxes = detections[:, :4]
	confidences = detections[:, 4]
	class_ids = detections[:, 5]

	# Confidence threshold 필터링
	filtered_indices = np.where(confidences >= confidence_threshold)[0]
	bboxes = bboxes[filtered_indices]
	confidences = confidences[filtered_indices]
	class_ids = class_ids[filtered_indices]

	# Non-Maximum Suppression (NMS)
	nms_indices = non_maximum_suppression(bboxes, confidences, iou_threshold)
	bboxes = bboxes[nms_indices]
	confidences = confidences[nms_indices]
	class_ids = class_ids[nms_indices]

	red = 0
	white = 0
	yellow = 0
	flag = False
	# 당구대 내부에 있는 바운딩 박스만 필터링
	inside_indices = []
	for i, (bbox, class_id) in enumerate(zip(bboxes, class_ids)):
		# 꼭짓점 좌표 내에 bbox가 위치하는지
		if (corners[0][0] < (bbox[0] + bbox[2]) / 2 < corners[2][0]) and (corners[0][1] < (bbox[1] + bbox[3]) / 2 < corners[2][1]):
			inside_indices.append(i)
		if class_id == 0:
			red += 1
		elif class_id == 1:
			white += 1
		elif class_id == 2:
			yellow += 1

	if red == 2 and white == 1 and yellow == 1:
		flag = True
	# 결과 반환
	return flag, bboxes, confidences, class_ids
