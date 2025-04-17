import numpy as np
import cv2
import math

# 특정 좌표에서 degree 방향 단위벡터의 끝점
def direction(x, y, degree):
	rad = math.radians(degree)

	return x + math.cos(degree), y + math.sin(rad)

# bbox의 중심
def center_of_bbox(x1, y1, x2, y2):
	return np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float64)		# y2는 공의 맨 아래 y값을 나타냄. 공의 중심보다 맨 바닥으로 하는게 더 정확할 듯

# 당구대의 네 꼭짓점 (픽셀 좌표)

# 당구대 크기(단위 mm)
table_width = 600
table_height = 300

# 당구대의 world 좌표 (mm 단위)
table_world = np.array([
    [0, 0, 20],             # 왼쪽 위
    [table_width, 0, 20],   # 오른쪽 위
    [table_width, table_height, 20],  # 오른쪽 아래
    [0, table_height, 20]   # 왼쪽 아래
], dtype=np.float64)

# 카메라 Intrinsic Matrix
intrinsic_matrix = np.array([
	[627.61810921, 0, 336.60305213],
	[0, 627.41336184, 228.74216909],
	[0, 0, 1]], dtype=np.float64)

# 카메라 distort coefficients
dist_coeffs = np.array([0.00525704, 0.13089543, -0.00100527, -0.00176605, -0.65529096])

focus = 627

########################################################################################################################

# 두 점 사이 거리
def calculate_distance(point1, point2):
	return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 이미지 기준 좌표계 -> 당구대 기준 좌표계
def image_to_world(ball_image, table_image, table_world=table_world, intrinsic_matrix=intrinsic_matrix, distort_coeffs=dist_coeffs):
	# SolvePnP를 이용해 Rotation vector, Translation vector 추정
	_, rvec, tvec = cv2.solvePnP(table_world, table_image, intrinsic_matrix, distort_coeffs)

	# Z_world 계산: 카메라부터 공까지 실제 거리 추정
	z_world = tvec[2]


	# Rotation Vector를 Rotation Matrix로 변환
	R, _ = cv2.Rodrigues(rvec)
	t = tvec.reshape(3, 1)
	inv_R = np.linalg.inv(R)
	inv_K = np.linalg.inv(intrinsic_matrix)

	# 당구공의 픽셀 좌표 -> 월드 좌표 변환
	# Homogeneous 픽셀 좌표
	ball_image_homogeneous = np.array([ball_image[0], ball_image[1], 1], dtype=np.float64).reshape(3, 1)

	# 역 변환: 월드 좌표 계산
	mid_cal = z_world * np.dot(inv_K, ball_image_homogeneous) - t	# 중간 계산
	ball_world = np.dot(inv_R, mid_cal)

	return ball_world

# 당구대 기준 좌표계 -> 이미지 기준 좌표계
def world_to_image(ball_world, table_image, table_world=table_world, intrinsic_matrix=intrinsic_matrix, distort_coeffs=dist_coeffs):
	# SolvePnP를 이용해 Rotation vector, Translation vector 추정
	_, rvec, tvec = cv2.solvePnP(table_world, table_image, intrinsic_matrix, distort_coeffs)

	# Z_world 계산: 카메라부터 공까지 실제 거리 추정
	z_world = tvec[2]

	# 회전 벡터 -> 회전 행렬 변환
	R, _ = cv2.Rodrigues(rvec)
	t = tvec.reshape(3, 1)
	K = intrinsic_matrix
	ball_world = ball_world.reshape(3, 1)


	# 월드 좌표 -> 이미지 좌표 (3D -> 2D)
	image_points, _ = cv2.projectPoints(ball_world.reshape(-1, 3), rvec, tvec, intrinsic_matrix, distort_coeffs)

	image_points = image_points.ravel()


	return int(image_points[0]), int(image_points[1])