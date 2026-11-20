
import cv2
import os
import glob
import numpy as np

from face_recognition_engine import FaceRecognitionEngine
from attendance_manager import AttendanceManager
from config import (
    KNOWN_FACES_DIR,
    SIMILARITY_THRESHOLD,
    REQUIRED_CONSECUTIVE_FRAMES,
    TOP_K_MATCH_SCORES,
    EMBEDDING_DIM,
)


def load_registered_students():
    students = {}

    files = glob.glob(
        os.path.join(KNOWN_FACES_DIR, "*.npz")
    )

    for file_path in files:
        data = np.load(file_path, allow_pickle=False)

        name = str(data["name"].item())
        roll_number = str(data["roll_number"].item())

        embeddings = data["embeddings"].astype(np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            print("Skipping invalid embedding file:", file_path)
            continue

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = embeddings / np.maximum(norms, 1e-12)

        students[roll_number] = {
            "name": name,
            "embeddings": embeddings
        }

    return students


def compare_embedding(query_embedding, registered_embeddings):
    query_embedding = query_embedding / max(
        np.linalg.norm(query_embedding),
        1e-12
    )

    scores = registered_embeddings @ query_embedding

    top_scores = np.sort(scores)[-min(TOP_K_MATCH_SCORES, len(scores)):]

    return float(np.mean(top_scores))


def find_best_match(embedding, students):
    best_match = None
    best_score = -1.0

    for roll_number, student in students.items():
        score = compare_embedding(
            embedding,
            student["embeddings"]
        )

        if score > best_score:
            best_score = score

            best_match = {
                "name": student["name"],
                "roll_number": roll_number,
                "similarity": score
            }

    return best_match


def main():
    engine = FaceRecognitionEngine()
    attendance_manager = AttendanceManager()

    students = load_registered_students()

    if not students:
        print("No registered students found.")
        return

    print("\nRegistered students:")

    for roll_number, student in students.items():
        print(
            f"{student['name']} - {roll_number} "
            f"({len(student['embeddings'])} samples)"
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    candidate_roll = None
    consecutive_frames = 0

    attendance_status = "Waiting for recognition"

    # Tracks students whose attendance has already been processed
    # during this webcam session.
    attendance_processed = set()

    print("\nLive recognition and attendance started.")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            results = engine.process_frame(frame)

            if len(results) != 1:
                candidate_roll = None
                consecutive_frames = 0

                if len(results) == 0:
                    attendance_status = "No face detected"
                else:
                    attendance_status = "Multiple faces detected"

            for result in results:
                x, y, w, h = result["bbox"]

                match = find_best_match(
                    result["embedding"],
                    students
                )

                if match is None:
                    continue

                score = match["similarity"]

                if score >= SIMILARITY_THRESHOLD and len(results) == 1:

                    roll_number = match["roll_number"]

                    if candidate_roll == roll_number:
                        consecutive_frames += 1
                    else:
                        candidate_roll = roll_number
                        consecutive_frames = 1

                    if roll_number not in attendance_processed:

                        attendance_status = (
                            f"Confirming: {consecutive_frames}/"
                            f"{REQUIRED_CONSECUTIVE_FRAMES}"
                        )

                        if consecutive_frames >= REQUIRED_CONSECUTIVE_FRAMES:

                            response = attendance_manager.mark_attendance(
                                match["name"],
                                roll_number
                            )

                            attendance_processed.add(roll_number)

                            if response["success"]:
                                attendance_status = (
                                    "ATTENDANCE MARKED SUCCESSFULLY"
                                )
                            else:
                                attendance_status = (
                                    "ATTENDANCE ALREADY MARKED TODAY"
                                )

                            print(
                                f"\n{attendance_status}"
                                f"\nName: {match['name']}"
                                f"\nRoll Number: {roll_number}"
                            )

                    else:
                        attendance_status = (
                            "ATTENDANCE ALREADY PROCESSED THIS SESSION"
                        )

                    label = (
                        f"{match['name']} ({roll_number})"
                    )

                    color = (0, 255, 0)

                else:
                    candidate_roll = None
                    consecutive_frames = 0

                    attendance_status = "UNKNOWN PERSON"
                    label = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Similarity: {score:.3f}",
                    (x, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            cv2.putText(
                frame,
                attendance_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "Face Recognition and Attendance",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
