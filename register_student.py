
import cv2
import os
import re
import numpy as np

from face_recognition_engine import FaceRecognitionEngine
from config import (
    KNOWN_FACES_DIR,
    SAMPLES_REQUIRED,
    ROLL_NUMBER_REGEX,
)


def validate_roll_number(roll_number):
    return re.fullmatch(ROLL_NUMBER_REGEX, roll_number) is not None


def register_student():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    name = input("Enter student name: ").strip()
    roll_number = input("Enter roll number: ").strip()

    if not name or not roll_number:
        print("Name and roll number cannot be empty.")
        return

    if not validate_roll_number(roll_number):
        print("Invalid roll number. Use letters, numbers, _ or -.")
        return

    save_path = os.path.join(
        KNOWN_FACES_DIR,
        f"{roll_number}.npz"
    )

    if os.path.exists(save_path):
        print("This roll number is already registered.")
        return

    engine = FaceRecognitionEngine()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    embeddings = []

    print("\nRegistration started.")
    print("Position your face in front of the camera.")
    print("Press SPACE to capture a sample.")
    print("Press Q to cancel.")
    print("Capture 10 samples with slight changes in angle.")

    try:
        while len(embeddings) < SAMPLES_REQUIRED:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            results = engine.process_frame(frame)

            for result in results:
                x, y, w, h = result["bbox"]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Samples: {len(embeddings)}/{SAMPLES_REQUIRED}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if len(results) == 1:
                message = "One face detected - Press SPACE"
            elif len(results) == 0:
                message = "No face detected"
            else:
                message = "Multiple faces detected - Only one allowed"

            cv2.putText(
                frame,
                message,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("Student Registration", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Registration cancelled.")
                return

            if key == 32:
                if len(results) != 1:
                    print("Capture rejected: exactly one face is required.")
                    continue

                embedding = results[0]["embedding"]

                if not np.all(np.isfinite(embedding)):
                    print("Invalid embedding. Try again.")
                    continue

                embeddings.append(embedding)

                print(
                    f"Captured sample "
                    f"{len(embeddings)}/{SAMPLES_REQUIRED}"
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(embeddings) != SAMPLES_REQUIRED:
        print("Registration incomplete. No student data saved.")
        return

    embeddings = np.stack(embeddings).astype(np.float32)

    np.savez_compressed(
        save_path,
        name=np.array(name),
        roll_number=np.array(roll_number),
        embeddings=embeddings
    )

    print("\nStudent registered successfully!")
    print("Name:", name)
    print("Roll number:", roll_number)
    print("Embedding samples:", len(embeddings))
    print("Embedding shape:", embeddings.shape)
    print("Saved at:", save_path)


if __name__ == "__main__":
    register_student()
