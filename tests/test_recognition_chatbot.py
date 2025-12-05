import cv2
import face_recognition
import numpy as np


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        return

    print("[INFO] 웹캠이 켜졌습니다.")
    print("[INFO] 화면에 얼굴을 비추면 초록 박스로 표시하려고 시도합니다.")
    print("[INFO] r 키: 현재 프레임에서 얼굴 인코딩 시도")
    print("[INFO] q 키: 프로그램 종료")

    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] 프레임을 읽을 수 없습니다. 계속 시도합니다...")
            continue

        # === 핵심: dlib이 좋아하는 형태로 맞추기 ===
        # frame: BGR, uint8, (H, W, 3)
        frame = frame.astype("uint8", copy=False)

        # BGR -> RGB (공식 예제 스타일)
        rgb = frame[:, :, ::-1].copy()  # copy()로 contiguous 보장

        if first_frame:
            print("[DEBUG] frame dtype:", frame.dtype, "shape:", frame.shape)
            print("[DEBUG] rgb dtype:", rgb.dtype, "shape:", rgb.shape)
            print("[DEBUG] rgb flags:", rgb.flags)  # C_CONTIGUOUS 확인용
            first_frame = False

        # 얼굴 위치 찾기 (hog 모델)
        boxes = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=0,  # 일단 0으로, 나중에 1~2로 조정 가능
            model="hog"
        )

        face_count = len(boxes)

        # 얼굴 박스 그리기
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        info_text = f"Faces: {face_count} | r: encode, q: quit"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if face_count == 0:
                print("[INFO] 현재 프레임에서 얼굴을 찾지 못했습니다. 카메라와의 거리/조명을 조정해 보세요.")
            else:
                encodings = face_recognition.face_encodings(rgb, boxes)
                print(f"[INFO] 감지된 얼굴 개수: {len(encodings)}")
                if len(encodings) > 0:
                    print("[INFO] 첫 번째 얼굴 인코딩 벡터 길이:", len(encodings[0]))
                else:
                    print("[WARN] 얼굴 위치는 찾았지만 인코딩에 실패했습니다.")

        if key == ord('q'):
            print("[INFO] 프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
