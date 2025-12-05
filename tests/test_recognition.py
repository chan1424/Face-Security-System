import cv2
import face_recognition
import numpy as np


def load_authorized_encoding(image_path: str):
    image = face_recognition.load_image_file("me_camera.jpg")
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        raise ValueError("[ERROR] 인가된 얼굴 이미지에서 얼굴을 찾지 못했습니다.")
    if len(encodings) > 1:
        print("[WARN] 인가된 이미지에서 여러 얼굴을 찾았습니다. 첫 번째 얼굴만 사용합니다.")

    return encodings[0]


def main():
    AUTH_IMAGE_PATH = "authorized.jpg"

    try:
        authorized_encoding = load_authorized_encoding(AUTH_IMAGE_PATH)
        print("[INFO] 인가된 사용자 얼굴 인코딩 로드 완료")
    except Exception as e:
        print(e)
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    # === 카메라 FPS 정보 찍어보기 (아래에서 설명할 부분) ===
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 카메라에서 보고하는 FPS: {fps}")

    print("[INFO] 인가된 사용자 1명만 있을 때: UNLOCK, 그 외: LOCK")
    print("[INFO] q 키: 종료")

    first_frame = True
    frame_index = 0
    CHECK_INTERVAL = 10  # 10프레임마다 한 번씩 얼굴 인식

    # 마지막 인식 결과 저장용
    last_boxes = []
    last_labels = []
    last_state = "LOCK"
    last_face_count = 0
    last_authorized_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] 프레임을 읽을 수 없습니다. 계속 시도합니다...")
            continue

        frame = frame.astype("uint8", copy=False)
        rgb = frame[:, :, ::-1].copy()

        if first_frame:
            print("[DEBUG] frame dtype:", frame.dtype, "shape:", frame.shape)
            print("[DEBUG] rgb dtype:", rgb.dtype, "shape:", rgb.shape)
            print("[DEBUG] rgb flags:", rgb.flags)
            first_frame = False

        frame_index += 1

        # ========================
        # 10프레임마다 한 번씩만 얼굴 인식 수행
        # ========================
        if frame_index % CHECK_INTERVAL == 0:
            boxes = face_recognition.face_locations(
                rgb,
                number_of_times_to_upsample=0,
                model="hog"
            )
            encodings = face_recognition.face_encodings(rgb, boxes)

            face_count = len(boxes)
            authorized_match_count = 0
            labels = []

            for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
                matches = face_recognition.compare_faces(
                    [authorized_encoding],
                    face_encoding,
                    tolerance=0.4
                )
                distance = face_recognition.face_distance(
                    [authorized_encoding],
                    face_encoding
                )[0]

                if matches[0]:
                    authorized_match_count += 1
                    label = f"AUTHORIZED ({distance:.2f})"
                    labels.append((label, (top, right, bottom, left), True))
                else:
                    label = f"UNKNOWN ({distance:.2f})"
                    labels.append((label, (top, right, bottom, left), False))

            # 상태 로직 업데이트
            if face_count == 1 and authorized_match_count == 1:
                state = "UNLOCK"
            else:
                state = "LOCK"

            # 결과를 last_*에 저장
            last_boxes = boxes
            last_labels = labels
            last_state = state
            last_face_count = face_count
            last_authorized_count = authorized_match_count

        # ========================
        # 여기서는 마지막 결과(last_*)를 사용해서 화면만 그림
        # ========================
        for label, (top, right, bottom, left), is_authorized in last_labels:
            color = (0, 255, 0) if is_authorized else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 상태 텍스트
        state_color = (0, 255, 0) if last_state == "UNLOCK" else (0, 0, 255)
        info_text = f"Faces: {last_face_count}, Authorized: {last_authorized_count}, STATE: {last_state}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        cv2.imshow("Face Security Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
