import cv2
import face_recognition
import numpy as np

print("🚀 시스템 초기화 중...")

# 1. 웹캠 실행
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ 오류: 웹캠을 열 수 없습니다.")
    exit()

# 2. 카메라 해상도 강제 설정 (너무 크면 느리니까 적당히 조절)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 얼굴 탐지 시작... (종료: q)")

while True:
    ret, frame = video_capture.read()
    
    # 3. 프레임 유효성 검사 (아주 중요!)
    if not ret or frame is None:
        print("❌ 프레임을 읽지 못했습니다. (카메라 연결 확인 필요)")
        break

    try:
        # --- [진단 로그] ---
        # 실행 초기에 이미지 정보를 한 번 출력해봅니다.
        # 정상이라면 (480, 640, 3) uint8 같은 형태여야 합니다.
        # -------------------
        # print(f"Shape: {frame.shape}, Type: {frame.dtype}") 

        # 4. 리사이징 제거! (원본 그대로 사용)
        # 리사이징 과정에서 메모리 배열이 꼬이는 경우가 많습니다.
        # 일단 원본으로 테스트합니다.
        
        # 5. BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 6. [강력한 형변환]
        # np.array()를 다시 감싸면 데이터를 새로 복사하면서 메모리를 정렬합니다.
        # dtype=np.uint8을 명시해서 dlib이 좋아하는 형태로 만듭니다.
        clean_frame = np.array(rgb_frame, dtype=np.uint8)

        # 7. 얼굴 위치 찾기
        face_locations = face_recognition.face_locations(clean_frame)

        # 8. 그리기
        for (top, right, bottom, left) in face_locations:
            # 리사이징을 안 했으니 좌표 곱하기(*4)도 필요 없습니다.
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)

        cv2.imshow('Debug Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"\n⚠️ 치명적 오류 발생!")
        print(f"에러 메시지: {e}")
        print(f"현재 이미지 정보: {frame.shape if 'frame' in locals() else 'None'}, {frame.dtype if 'frame' in locals() else 'None'}")
        break

video_capture.release()
cv2.destroyAllWindows()