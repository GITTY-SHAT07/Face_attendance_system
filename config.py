import os
import re
from pathlib import Path
from dataclasses import dataclass


# ==============================================================================
# BASE DIRECTORIES & FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
CHECKPOINTS_DIR = Path("checkpoints")
TRAINING_LOGS_DIR = Path("training_logs")
KNOWN_FACES_DIR = "known_faces"
EXPORTS_DIR = Path("data/exports")

# Specific model & data file paths
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LANDMARKER_MODEL_PATH = "models/face_landmarker.task"
CHECKPOINT_PATH = "checkpoints/best_resnet.pth"
CLASS_MAPPING_PATH = "checkpoints/class_mapping.json"
ATTENDANCE_FILE = Path("data/attendance.csv")
EVALUATION_LOG_PATH = "training_logs/embedding_evaluation.json"
TRAINING_HISTORY_PATH = "training_logs/training_history.json"


# ==============================================================================
# FACE DETECTION (YuNet ONNX)
# ==============================================================================
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.80
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

# Face filter constraints
MIN_FACE_SIZE = 60          # Minimum width & height in pixels
MIN_ASPECT_RATIO = 0.55     # Width / Height minimum
MAX_ASPECT_RATIO = 1.50     # Width / Height maximum
FACE_CROP_PADDING = 0.15    # Padding around detected bounding box


# ==============================================================================
# ACTIVE LIVENESS & ANTI-SPOOFING (MediaPipe FaceMesh & SolvePnP)
# ==============================================================================
LIVENESS_MIN_DETECTION_CONFIDENCE = 0.6
LIVENESS_MIN_PRESENCE_CONFIDENCE = 0.6
LIVENESS_MIN_TRACKING_CONFIDENCE = 0.6

# Eye Aspect Ratio (EAR) for blink detection
EAR_THRESHOLD = 0.21
CLOSED_FRAMES_REQUIRED = 2

# Head pose yaw thresholds (in degrees)
HEAD_TURN_THRESHOLD = 12.0
HEAD_RETURN_THRESHOLD = 5.0
CALIBRATION_FRAMES = 25


# ==============================================================================
# FACE RECOGNITION ARCHITECTURE & PREPROCESSING (myResnet20)
# ==============================================================================
FACE_INPUT_SIZE = (112, 112)
EMBEDDING_DIM = 128
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


# ==============================================================================
# MATCHING & TEMPORAL VOTING
# ==============================================================================
SIMILARITY_THRESHOLD = 0.751
TOP_K_MATCH_SCORES = 1
REQUIRED_CONSECUTIVE_FRAMES = 5


# ==============================================================================
# STUDENT REGISTRATION
# ==============================================================================
REQUIRED_REGISTRATION_SAMPLES = 10
SAMPLES_REQUIRED = REQUIRED_REGISTRATION_SAMPLES
ROLL_NUMBER_REGEX = r"[A-Za-z0-9_-]+"


# ==============================================================================
# TRAINING & LOSS HYPERPARAMETERS (ArcFace)
# ==============================================================================
ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 32.0
DEFAULT_NUM_CLASSES = 40
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 60
TRAIN_LR = 0.001
TRAIN_WEIGHT_DECAY = 0.0005


# ==============================================================================
# WEB SERVER & CAMERA
# ==============================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
SERVER_DEBUG = False
CAMERA_INDEX = 0


# ==============================================================================
# UNIFIED CONFIG DATACLASS
# ==============================================================================
@dataclass(frozen=True)
class Config:
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    training_logs_dir: Path = TRAINING_LOGS_DIR
    known_faces_dir: str = KNOWN_FACES_DIR
    exports_dir: Path = EXPORTS_DIR

    yunet_model_path: str = YUNET_MODEL_PATH
    landmarker_model_path: str = LANDMARKER_MODEL_PATH
    checkpoint_path: str = CHECKPOINT_PATH
    class_mapping_path: str = CLASS_MAPPING_PATH
    attendance_file: Path = ATTENDANCE_FILE
    evaluation_log_path: str = EVALUATION_LOG_PATH
    training_history_path: str = TRAINING_HISTORY_PATH

    yunet_input_size: tuple = YUNET_INPUT_SIZE
    yunet_score_threshold: float = YUNET_SCORE_THRESHOLD
    yunet_nms_threshold: float = YUNET_NMS_THRESHOLD
    yunet_top_k: int = YUNET_TOP_K
    min_face_size: int = MIN_FACE_SIZE
    min_aspect_ratio: float = MIN_ASPECT_RATIO
    max_aspect_ratio: float = MAX_ASPECT_RATIO
    face_crop_padding: float = FACE_CROP_PADDING

    liveness_min_detection_confidence: float = LIVENESS_MIN_DETECTION_CONFIDENCE
    liveness_min_presence_confidence: float = LIVENESS_MIN_PRESENCE_CONFIDENCE
    liveness_min_tracking_confidence: float = LIVENESS_MIN_TRACKING_CONFIDENCE
    ear_threshold: float = EAR_THRESHOLD
    closed_frames_required: int = CLOSED_FRAMES_REQUIRED
    head_turn_threshold: float = HEAD_TURN_THRESHOLD
    head_return_threshold: float = HEAD_RETURN_THRESHOLD
    calibration_frames: int = CALIBRATION_FRAMES

    face_input_size: tuple = FACE_INPUT_SIZE
    embedding_dim: int = EMBEDDING_DIM
    norm_mean: tuple = NORM_MEAN
    norm_std: tuple = NORM_STD

    similarity_threshold: float = SIMILARITY_THRESHOLD
    top_k_match_scores: int = TOP_K_MATCH_SCORES
    required_consecutive_frames: int = REQUIRED_CONSECUTIVE_FRAMES

    required_registration_samples: int = REQUIRED_REGISTRATION_SAMPLES
    samples_required: int = SAMPLES_REQUIRED
    roll_number_regex: str = ROLL_NUMBER_REGEX

    arcface_margin: float = ARCFACE_MARGIN
    arcface_scale: float = ARCFACE_SCALE
    default_num_classes: int = DEFAULT_NUM_CLASSES
    train_batch_size: int = TRAIN_BATCH_SIZE
    train_epochs: int = TRAIN_EPOCHS
    train_lr: float = TRAIN_LR
    train_weight_decay: float = TRAIN_WEIGHT_DECAY

    server_host: str = SERVER_HOST
    server_port: int = SERVER_PORT
    server_debug: bool = SERVER_DEBUG
    camera_index: int = CAMERA_INDEX


CFG = Config()
