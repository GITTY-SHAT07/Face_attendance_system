
import cv2
import torch
import numpy as np
import os

from resnet_face import myResnet20
from face_detector import FaceDetector


CHECKPOINT_PATH = "checkpoints/best_resnet.pth"
EMBEDDING_DIM = 128


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FaceRecognitionEngine:

    def __init__(self):
        self.device = select_device()

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        self.model = myResnet20(
            embedding_dim=EMBEDDING_DIM
        ).to(self.device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.detector = FaceDetector()

        print("Face recognition engine initialized.")
        print("Device:", self.device)
        print("Checkpoint epoch:", checkpoint["epoch"])

    def preprocess_face(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        face_resized = cv2.resize(
            face_rgb,
            (112, 112),
            interpolation=cv2.INTER_AREA
        )

        face_array = face_resized.astype(
            np.float32
        ) / 255.0

        face_array = (face_array - 0.5) / 0.5

        face_array = np.transpose(
            face_array,
            (2, 0, 1)
        )

        face_tensor = torch.from_numpy(
            face_array
        ).unsqueeze(0)

        return face_tensor.to(self.device)

    def extract_embedding(self, face_crop):
        face_tensor = self.preprocess_face(face_crop)

        if face_tensor is None:
            return None

        with torch.inference_mode():
            embedding = self.model(face_tensor)

            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )

        return embedding.squeeze(0).cpu().numpy()

    def process_frame(self, frame):
        faces = self.detector.detect(frame)

        face_crops = self.detector.crop_faces(
            frame,
            faces
        )

        results = []

        for face, face_crop in zip(faces, face_crops):
            embedding = self.extract_embedding(face_crop)

            if embedding is None:
                continue

            x, y, w, h = map(int, face[:4])

            results.append({
                "bbox": (x, y, w, h),
                "confidence": float(face[14]),
                "embedding": embedding,
                "embedding_dimension": len(embedding)
            })

        return results


def main():
    engine = FaceRecognitionEngine()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    print("Live embedding extraction started.")
    print("Press Q to exit.")

    while True:
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

            label = (
                f"Embedding: {result['embedding_dimension']}D"
            )

            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces processed: {len(results)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "ResNet Face Embedding Extraction",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
