import cv2
import numpy as np
import time

from Detection.table_detect import *
from Detection.ball_detect import *
from Detection.coordinate_transform import *

from Solver.optimal_predict import find_optimal_angle

# 이미지 크기 480 * 640
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠, 다른 값으로 변경 가능

# 원하는 FPS 설정
desired_fps = 60
frame_duration = 1 / desired_fps  # 각 프레임의 목표 처리 시간 (초)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

class_names = {0: "Red Ball", 1: "White Ball", 2: "Yellow Ball"}
ball_image = {"Red1": None, "Red2": None, "White": None, "Yellow": None}		# 이미지 상에서의 공의 좌표 -> Red1, Red2, White, Yellow 순
ball_world = {"Red1": None, "Red2": None, "White": None, "Yellow": None}		# 당구대 좌표계 기준 공의 좌표 -> Red1, Red2, White, Yellow 순
optimal_angle = 0

while True:
	start_time = time.time()  # 프레임 처리 시작 시간

	ret, frame = cap.read()
	if not ret:
		print("프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")
		break

	# BGR 이미지를 HSV로 변환
	img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# 꼭짓점 좌표 찾기
	lines, corners = lines_and_corners(img_hsv)
	# print(len(corners))

	# 당구대의 모서리와 꼭짓점이 모두 탐지된 경우, 공 detection 시도
	# 공이 탐지된 적이 없는 경우. 공을 치면 ball_image = None으로 reset
	if len(lines) == 4 and len(corners) == 4:
		flag, bboxes, confidences, class_ids = detect_balls(frame, corners)

		# 공 4개가 모두 탐지된 경우
		if flag == True:
			# YOLO 결과 그리기
			ball_image['Red1'] = None
			for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
				# 공의 bbox 좌표
				x1, y1, x2, y2 = map(int, bbox)
				
				# 영상에 공 위치 표시 & 공의 label 출력
				label = f'{class_names.get(class_id)} {confidence:.2f}'
				cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 10, (0, 255, 0), 5)
				cv2.putText(frame, label, (x1, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				
				# 공의 좌표 저장(이미지 기준 & 당구대 기준 좌표계 모두)
				if class_id == 0:
					if ball_image['Red1'] is None:
						ball_image['Red1'] = center_of_bbox(x1, y1, x2, y2)
						ball_world['Red1'] = image_to_world(ball_image['Red1'], corners)
					else:
						ball_image['Red2'] = center_of_bbox(x1, y1, x2, y2)
						ball_world['Red2'] = image_to_world(ball_image['Red2'], corners)
				elif class_id == 1:
					ball_image['White'] = center_of_bbox(x1, y1, x2, y2)
					ball_world['White'] = image_to_world(ball_image['White'], corners)
				elif class_id == 2:
					ball_image['Yellow'] = center_of_bbox(x1, y1, x2, y2)
					ball_world['Yellow'] = image_to_world(ball_image['Yellow'], corners)
			print(f"White: ({ball_world['White'][0]} , {ball_world['White'][1]})")
			print(f"Yellow: ({ball_world['Yellow'][0]} , {ball_world['Yellow'][1]})")
			print(f"Red1: ({ball_world['Red1'][0]} , {ball_world['Red1'][1]})")
			print(f"Red2: ({ball_world['Red2'][0]} , {ball_world['Red2'][1]})")
			balls = [
				ball_world["White"][:2],
				ball_world['Yellow'][:2],
				ball_world['Red1'][:2],
				ball_world['Red2'][:2],
			]

			# 공의 당구대 기준 좌표계로 변환된 좌표를 이용하여 해 계산
			if optimal_angle == 0:
				optimal_angle = find_optimal_angle(balls)
				optimal_angle = optimal_angle
			print(f"optimal angle: {np.degrees(optimal_angle)}")

			# 계산된 해를 기반으로 공을 쳐야할 방향을 화살표로 시각화
			goal = direction(ball_world['Yellow'][0], ball_world['Yellow'][1], optimal_angle)	# 칠 방향의 단위벡터 지점(world 좌표)
			goal = np.array([goal[0], goal[1], ball_world['Yellow'][2]], dtype=np.float32)
			# goal의 당구대 기준 좌표계를 이미지 기준 좌표로 변환
			image_goal = world_to_image(goal, corners)
			cv2.arrowedLine(frame, (int(ball_image['Yellow'][0]), int(ball_image['Yellow'][1])), (int(image_goal[0]), int(image_goal[1])), (0, 255, 0), 2, cv2.LINE_AA)
		else:
			print("공 4개를 보여주세요!")
			pass
	else:
		print("네 모서리를 모두 보여주세요!")
		pass

	# 당구대의 꼭짓점 & 모서리 시각화
	if lines is not None:
		for rho, theta, foot_x, foot_y in lines:
			# 직선 방정식 계산
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho

			# 직선의 시작과 끝 좌표 계산
			pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
			pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))

			# 직선 그리기
			cv2.line(frame, pt1, pt2, (255, 0, 0), 5)

	if corners is not None:
		for x, y in corners:
			cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 5)  # 초록색 점으로 표시

    # 결과 표시
	cv2.imshow("Original Frame", frame)

	# 녹화
	# out.write(frame)

	# FPS 제어: 남은 시간 동안 대기
	elapsed_time = time.time() - start_time
	delay = max(0, frame_duration - elapsed_time)  # 처리 시간 이후 남은 시간 계산
	time.sleep(delay)

	key = cv2.waitKey(1) & 0xFF
    # ESC 키를 누르면 종료
	if key == 27:
		break
	# c 누르면 재계산
	elif key == ord('c'):
		optimal_angle = 0
		print("Recalculate")

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
