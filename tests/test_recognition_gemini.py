import cv2
import face_recognition
import numpy as np # numpy가 꼭 필요합니다

# 1. 웹캠 실행
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ 오류: 웹캠을 열 수 없습니다.")
    exit()

print("🎥 얼굴 탐지 시작... (종료: q)")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # 2. 이미지 리사이징 (속도 최적화)
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # 3. 색상 변환 (BGR -> RGB)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # ✨ [핵심 수정] Dlib을 위한 강제 형변환 ✨
        # Dlib은 엄격하게 'uint8' 타입과 'C-contiguous' 메모리 구조를 원합니다.
        # 아래 한 줄이 에러를 해결해 줄 겁니다.
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

        # (디버깅용) 만약 에러가 계속나면 이 정보가 필요합니다.
        # print(f"Shape: {rgb_small_frame.shape}, Type: {rgb_small_frame.dtype}")

        # 4. 얼굴 위치 찾기
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # 5. 그리기
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, "Face", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Detection Fixed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"⚠️ 실행 중 에러 발생: {e}")
        break

video_capture.release()
cv2.destroyAllWindows()