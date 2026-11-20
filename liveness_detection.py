
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import random

from config import (
    LANDMARKER_MODEL_PATH,
    LIVENESS_MIN_DETECTION_CONFIDENCE,
    LIVENESS_MIN_PRESENCE_CONFIDENCE,
    LIVENESS_MIN_TRACKING_CONFIDENCE,
    EAR_THRESHOLD,
    CLOSED_FRAMES_REQUIRED,
    HEAD_TURN_THRESHOLD,
    HEAD_RETURN_THRESHOLD,
    CALIBRATION_FRAMES,
)


class LivenessDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or LANDMARKER_MODEL_PATH

        if not os.path.exists(self.model_path):
            # Fallback to relative to this file
            fallback = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")
            if os.path.exists(fallback):
                self.model_path = fallback
            else:
                raise FileNotFoundError(self.model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=LIVENESS_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=LIVENESS_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=LIVENESS_MIN_TRACKING_CONFIDENCE
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        self.left_eye = [362, 385, 387, 263, 373, 380]
        self.right_eye = [33, 160, 158, 133, 153, 144]

        self.ear_threshold = EAR_THRESHOLD
        self.closed_frames_required = CLOSED_FRAMES_REQUIRED

        self.turn_threshold = HEAD_TURN_THRESHOLD
        self.return_threshold = HEAD_RETURN_THRESHOLD
        self.calibration_frames = CALIBRATION_FRAMES

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        self.pose_indices = [1, 152, 33, 263, 61, 291]

        self.last_timestamp = 0

        self.reset()

    def reset(self):
        self.blink_count = 0
        self.closed_frames = 0
        self.eye_was_closed = False

        self.state = "BLINK"

        self.center_samples = []
        self.center_yaw = None

        self.challenge = random.sample(
            ["LEFT", "RIGHT"], 2
        )

        self.challenge_index = 0
        self.completed = False

    def calculate_ear(self, landmarks, indices):
        points = np.array([
            [landmarks[i].x, landmarks[i].y]
            for i in indices
        ])

        vertical_1 = np.linalg.norm(
            points[1] - points[5]
        )

        vertical_2 = np.linalg.norm(
            points[2] - points[4]
        )

        horizontal = np.linalg.norm(
            points[0] - points[3]
        )

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    def estimate_yaw(self, landmarks, width, height):
        image_points = np.array([
            (
                landmarks[i].x * width,
                landmarks[i].y * height
            )
            for i in self.pose_indices
        ], dtype=np.float64)

        focal_length = width
        center = (width / 2, height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        return float(angles[1])

    def process_frame(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        timestamp = time.monotonic_ns() // 1_000_000

        timestamp = max(
            timestamp,
            self.last_timestamp + 1
        )

        self.last_timestamp = timestamp

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        status = {
            "face_detected": False,
            "ear": 0.0,
            "blink_count": self.blink_count,
            "state": self.state,
            "liveness_detected": self.completed
        }

        if not result.face_landmarks:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            self.closed_frames = 0
            self.eye_was_closed = False

            return frame, status

        landmarks = result.face_landmarks[0]

        height, width = frame.shape[:2]

        left_ear = self.calculate_ear(
            landmarks,
            self.left_eye
        )

        right_ear = self.calculate_ear(
            landmarks,
            self.right_eye
        )

        ear = (left_ear + right_ear) / 2.0

        yaw = self.estimate_yaw(
            landmarks,
            width,
            height
        )

        status["face_detected"] = True
        status["ear"] = ear
        status["yaw"] = yaw

        for index in self.left_eye + self.right_eye:
            x = int(landmarks[index].x * width)
            y = int(landmarks[index].y * height)

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        # Stage 1: Blink verification
        if self.state == "BLINK":

            if ear < self.ear_threshold:
                self.closed_frames += 1

                if (
                    self.closed_frames
                    >= self.closed_frames_required
                ):
                    self.eye_was_closed = True

            else:
                if self.eye_was_closed:
                    self.blink_count += 1
                    self.eye_was_closed = False

                self.closed_frames = 0

            if self.blink_count >= 1:
                self.state = "CALIBRATING"

        # Stage 2: Neutral head calibration
        elif self.state == "CALIBRATING":

            if yaw is not None:
                self.center_samples.append(yaw)

            if len(self.center_samples) >= self.calibration_frames:
                self.center_yaw = float(
                    np.mean(self.center_samples)
                )

                self.state = "TURN"

        # Stage 3: Random head movement
        elif self.state == "TURN":

            if (
                yaw is not None
                and self.center_yaw is not None
            ):
                difference = yaw - self.center_yaw

                direction = self.challenge[
                    self.challenge_index
                ]

                expected_sign = (
                    -1 if direction == "LEFT" else 1
                )

                if (
                    expected_sign * difference
                    >= self.turn_threshold
                ):
                    self.state = "RETURN"

        # Stage 4: Return to neutral position
        elif self.state == "RETURN":

            if (
                yaw is not None
                and self.center_yaw is not None
            ):
                difference = yaw - self.center_yaw

                if abs(difference) <= self.return_threshold:
                    self.challenge_index += 1

                    if self.challenge_index >= len(
                        self.challenge
                    ):
                        self.state = "PASSED"
                        self.completed = True
                    else:
                        self.state = "TURN"

        # Display eye and head information
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}  Blinks: {self.blink_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        if yaw is not None:
            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

        if self.state == "BLINK":
            message = "Please blink"

        elif self.state == "CALIBRATING":
            message = "Keep your head straight"

        elif self.state == "TURN":
            message = (
                f"Turn your head "
                f"{self.challenge[self.challenge_index]}"
            )

        elif self.state == "RETURN":
            message = "Return to center"

        else:
            message = "LIVENESS VERIFIED"

        color = (
            (0, 255, 0)
            if self.completed
            else (0, 165, 255)
        )

        cv2.putText(
            frame,
            message,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

        status["blink_count"] = self.blink_count
        status["state"] = self.state
        status["liveness_detected"] = self.completed

        return frame, status

    def close(self):
        self.landmarker.close()


def main():
    detector = LivenessDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not access webcam.")
        detector.close()
        return

    print("Liveness verification started.")
    print("Follow the instructions shown on screen.")
    print("Press Q to exit.")

    previous_blink_count = 0
    previous_state = None

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            frame, status = detector.process_frame(frame)

            cv2.imshow(
                "Face Liveness Detection",
                frame
            )

            if status["blink_count"] > previous_blink_count:
                previous_blink_count = status["blink_count"]
                print("Blink detected!")

            if status["state"] != previous_state:
                previous_state = status["state"]
                print("Liveness stage:", previous_state)

            if status["liveness_detected"]:
                print("LIVENESS VERIFICATION PASSED")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
