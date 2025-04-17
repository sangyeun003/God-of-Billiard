# [당구대 정보를 추출하는 함수들]
# 1. 당구대 영역 검출 -> detect_green(), fill_hole_by_balls(), extract_biggest_blob_with_center()
# 2. 당구대 모서리 & 꼭짓점 검출 -> vector_degree(), detect_table_edge(), sort_corners_clockwise(), detect_corners()
# 3. 종합 -> lines_and_corners()
import numpy as np
import cv2
from matplotlib import pyplot as plt

def detect_green(img_hsv):
    lower_green = np.array([40, 100, 100])  # HSV 하한 [50, 40, 50]
    upper_green = np.array([90, 255, 255])  # HSV 상한 [85, 255, 255]

    # 파란색 감지
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

    return green_mask

# 당구공으로 인해 생긴 구멍을 막아줌
def fill_hole_by_balls(input_image):
	# 커널 사이즈
	H_Size = 80

	# 테두리 추가
	# 모폴로지 연산을 위해 이미지 확장
	border = cv2.copyMakeBorder(input_image, H_Size+1, H_Size+1, H_Size+1, H_Size+1, 
								cv2.BORDER_CONSTANT, value=0)

	# 모폴로지 닫기 연산 -> 구멍을 메움
	# 커널 생성
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (H_Size, H_Size))
	# 구멍 메움
	morph = cv2.morphologyEx(border, cv2.MORPH_CLOSE, kernel)


	# 구멍 영역만 추출
	subtract = cv2.subtract(morph, border)

	# 침식 연산 및 경계 추출
	# 어두운 부분의 noise 제거
	eroded = cv2.erode(subtract, np.ones((3, 3), np.uint8), iterations=1)

	# 공 영역의 테두리만 땀
	edge = cv2.subtract(subtract, eroded)

	# 관심 영역 추출
	# copyMakeBorder로 확장한 부분을 제외한 원본 크기만 roi로 저장
	subtract_roi = subtract[H_Size+1:H_Size+1+input_image.shape[0], H_Size+1:H_Size+1+input_image.shape[1]]
	balls = subtract_roi

	# 당구공이 있는 위치를 매운 새로운 blue mask
	filled_billiard = cv2.add(balls, input_image)

	return balls, filled_billiard	# 구멍부분과 채워진 당구대부분 return

# 가장 큰 blob 찾고, 그 blob의 중심 구함
def extract_biggest_blob_with_center(input_image):
    # 연결 성분 계산
    # 영역 단위로 분석
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(input_image, connectivity=4)
    # label: 객체 별로 번호 지정
    # stats: N*5 행렬. N = 객체수+1(배경 때문에)
    # centroids: 각 영역의 무게중심 좌표

    # 가장 큰 블롭 찾기
    max_area = 0
    max_area_index = -1
    for i in range(1, num_labels):  # 0번 레이블은 배경
        area = stats[i, cv2.CC_STAT_AREA]	# 영역이 가장 큰 것을 찾음
        if area > max_area:
            max_area = area
            max_area_index = i

    # 중심 좌표 계산
    center = (int(centroids[max_area_index][0]), int(centroids[max_area_index][1]))

    # 가장 큰 블롭 마스크 생성
    biggest_blob = (labels == max_area_index).astype(np.uint8)
    return biggest_blob, center

################################################## 위는 당구대 영역 찾기 ##################################################
################################################### 아래는 모서리 찾기 ###################################################

# 모서리 detection

# 두 vector 간 각도 구하기
# 내적 이용
def vector_degree(x1, x2, y1, y2):
    dot_product = x1 * x2 + y1 * y2
    magnitude1 = np.sqrt(x1**2 + y1**2)
    magnitude2 = np.sqrt(x2**2 + y2**2)

    if magnitude1 * magnitude2 == 0:
        return np.inf  # 방지: 벡터 크기가 0인 경우

    cos_theta = dot_product / (magnitude1 * magnitude2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# [당구대 모서리를 찾기 위한 함수]
def detect_table_edge(big_blob, big_blob_center):
    # 침식 연산으로 경계 추출
    # 침식(erode) 연산하면 영역이 좀 작아짐
    # 큰거 - 작은거 = edge
    eroded = cv2.erode(big_blob, np.ones((3, 3), np.uint8), iterations=1)
    edge = cv2.subtract(big_blob, eroded)

    # 허프 변환으로 직선 검출
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 100)

    candidate_lines_info = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]  # rho와 theta 값 추출

            # 수직선의 안정성을 위해 sin(theta) 처리
            sine = np.sin(theta)
            if abs(sine) < 1e-4:
                sine += 1e-4
            
            # 직선 방정식의 계수 계산 (ax + by + c = 0 형태)
            a = -np.cos(theta) / sine
            b = -1.0
            c = rho / sine

            # 중심점에서 직선까지 수선의 발 계산
            p, q = big_blob_center
            foot = -(a * p + b * q + c) / (a**2 + b**2 + 1e-8)
            foot_x = foot * a + p
            foot_y = foot * b + q

            # 기존 직선들과 비교하여 중복 여부 확인
            dup = False
            for existing_line in candidate_lines_info:
                existing_rho, existing_theta, existing_foot_x, existing_foot_y = existing_line

                # 중심점 기준 직선의 방향과 각도 비교
                c_x = existing_foot_x - p
                c_y = existing_foot_y - q
                x = foot_x - p
                y = foot_y - q

				# blob의 중심까지의 수선의 발 간의 각도
                rad = vector_degree(x, c_x, y, c_y)
                
				# 두 직선 간 각도
                diff_theta = abs(existing_theta - theta) if abs(existing_theta - theta) < np.pi / 2 else np.pi - abs(existing_theta - theta)

                if diff_theta < 0.17 and rad < 0.5:  # 비슷한 직선으로 판단
                    dup = True
                    break
            
            # 중복되지 않은 직선만 후보에 추가
            if not dup:
                candidate_lines_info.append((rho, theta, foot_x, foot_y))

    return candidate_lines_info

# [당구대 꼭짓점을 찾기 위한 함수]
# corner들의 각도를 통해 시계방향으로 정렬
# x+y값이 가장 작은 점이 맨 처음에 옴 -> [좌상->우상->우하->좌하] 순
def sort_corners_clockwise(corners):
	if len(corners) != 0:
		# 코너의 중심 계산
		center_x = sum([x for x, y in corners]) / len(corners)
		center_y = sum([y for x, y in corners]) / len(corners)

		# 각도를 기준으로 시계방향 정렬
		sorted_corners = sorted(corners, key=lambda point: np.arctan2(point[1] - center_y, point[0] - center_x))

		# x + y가 가장 작은 점 찾기
		min_index = min(range(len(sorted_corners)), key=lambda i: sorted_corners[i][0] + sorted_corners[i][1])

		# 가장 작은 점을 리스트의 맨 앞으로 이동
		sorted_corners = sorted_corners[min_index:] + sorted_corners[:min_index]
		return sorted_corners
	return []

def detect_corners(lines, frame_size):
	corners = []

	for i in range(len(lines)):
		rho1, theta1, _, _ = lines[i]
		for j in range(i + 1, len(lines)):
			rho2, theta2, _, _ = lines[j]

			# 직선 간 교점 계산
			A = np.array([
				[np.cos(theta1), np.sin(theta1)],
				[np.cos(theta2), np.sin(theta2)]
			])
			b = np.array([rho1, rho2])

			# invertible한지 check
			if np.linalg.cond(A) < 1 / 1e-10:
				sol = np.linalg.solve(A, b)
				x, y = int(sol[0]), int(sol[1])
				if 0 <= x < frame_size[1] and 0 <= y < frame_size[0]:
					corners.append((int(sol[0]), int(sol[1])))
	corners = sort_corners_clockwise(list(set(corners)))
	return np.array(corners, dtype=np.float32)

##############################################################################

# [모서리 찾기 & 꼭짓점 찾기 종합]
def lines_and_corners(hsv):
	green_mask = detect_green(hsv)
	balls, filled_billiard = fill_hole_by_balls(green_mask)
	big, center = extract_biggest_blob_with_center(filled_billiard)
	
	lines = detect_table_edge(big, center)
	if len(lines) == 4:
		corners = detect_corners(lines, hsv.shape[:2])
	else:
		return [], []	# 길이가 0인 list
	return lines, corners