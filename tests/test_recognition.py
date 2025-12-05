# test_image_face_rec.py
import face_recognition

image = face_recognition.load_image_file("me_camera.jpg")
print("image dtype:", image.dtype, "shape:", image.shape)

boxes = face_recognition.face_locations(image, model="hog")
print("found faces:", len(boxes))
print("boxes:", boxes)
