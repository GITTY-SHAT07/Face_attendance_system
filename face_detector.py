
import cv2
import os


MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


class FaceDetector:

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self.detector = cv2.FaceDetectorYN_create(
            model_path,
            "",
            (320, 320),
            0.80,
            0.30,
            5000
        )

    def detect(self, frame):
        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        valid_faces = []

        for face in faces:
            x, y, w, h = face[:4]
            confidence = face[14]

            if confidence < 0.80:
                continue

            if w < 60 or h < 60:
                continue

            aspect_ratio = w / h

            if aspect_ratio < 0.55 or aspect_ratio > 1.5:
                continue

            valid_faces.append(face)

        return valid_faces

    def crop_faces(self, frame, faces, padding=0.15):
        height, width = frame.shape[:2]
        crops = []

        for face in faces:
            x, y, w, h = face[:4]

            pad_x = int(w * padding)
            pad_y = int(h * padding)

            x1 = max(0, int(x) - pad_x)
            y1 = max(0, int(y) - pad_y)
            x2 = min(width, int(x + w) + pad_x)
            y2 = min(height, int(y + h) + pad_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                crops.append(face_crop)

        return crops


def main():
    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    print("YuNet face detection started.")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        faces = detector.detect(frame)

        for face in faces:
            x, y, w, h = map(int, face[:4])
            confidence = face[14]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Face: {confidence:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("YuNet Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
