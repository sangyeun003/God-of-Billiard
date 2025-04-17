# c를 누를 때마다 사진을 capture_{timestamp}.jpg로 저장하는 프로그램
# q 누르면 종료
import cv2
import os
import time

# 저장할 디렉토리 설정
save_dir = "captured_frames"
os.makedirs(save_dir, exist_ok=True)

# 비디오 캡처 객체 생성 (기본 카메라 사용)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Press 'c' to capture an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera.")
        break

    # 실시간 영상 표시
    cv2.imshow("Live Camera", frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF  # 키 입력 읽기 (1ms 대기)

    if key == ord('c'):  # 'c' 키를 누르면 캡처
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured and saved: {filename}")

    elif key == ord('q'):  # 'q' 키를 누르면 종료
        print("Exiting")
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
