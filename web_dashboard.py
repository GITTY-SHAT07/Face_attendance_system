
import cv2
import csv
import glob
import os
import re
import io
import threading
import numpy as np

from pathlib import Path
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    render_template,
    Response
)

from face_recognition_engine import FaceRecognitionEngine
from attendance_manager import AttendanceManager
from liveness_detection import LivenessDetector


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

from config import (
    ATTENDANCE_FILE,
    KNOWN_FACES_DIR,
    SIMILARITY_THRESHOLD,
    REQUIRED_CONSECUTIVE_FRAMES,
    REQUIRED_REGISTRATION_SAMPLES,
    TOP_K_MATCH_SCORES,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_DEBUG,
)

app = Flask(__name__)


# --------------------------------------------------
# GLOBAL OBJECTS
# --------------------------------------------------

engine = None
students = {}
liveness_detector = None
liveness_verified = False

camera = None

camera_lock = threading.Lock()
registration_lock = threading.Lock()

attendance_manager = AttendanceManager()

attendance_processed = set()

candidate_roll = None
consecutive_frames = 0

live_status = {
    "running": False,
    "message": "Camera is stopped",
    "name": None,
    "roll_number": None,
    "similarity": None,
    "liveness_state": "NOT_STARTED",
    "liveness_verified": False
}

registration_state = {
    "active": False,
    "name": "",
    "roll_number": "",
    "embeddings": [],
    "capture_requested": False,
    "message": "Registration inactive"
}


# --------------------------------------------------
# INITIALIZE RECOGNITION ENGINE
# --------------------------------------------------

def initialize_recognition():

    global engine, students, liveness_detector

    if engine is None:
        engine = FaceRecognitionEngine()

    if liveness_detector is None:
        liveness_detector = LivenessDetector()
        liveness_detector.reset()

    if not students:

        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        files = glob.glob(
            os.path.join(KNOWN_FACES_DIR, "*.npz")
        )

        for file_path in files:

            try:
                with np.load(file_path, allow_pickle=False) as data:

                    name = str(data["name"].item())
                    roll = str(data["roll_number"].item())

                    embeddings = data["embeddings"].astype(
                        np.float32
                    )

                if embeddings.ndim != 2 or embeddings.shape[1] != 128:
                    print("Invalid embedding file:", file_path)
                    continue

                norms = np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True
                )

                embeddings = embeddings / np.maximum(
                    norms,
                    1e-12
                )

                students[roll] = {
                    "name": name,
                    "embeddings": embeddings
                }

            except Exception as error:
                print("Could not load:", file_path, error)

    return engine, students


# --------------------------------------------------
# ATTENDANCE DATA
# --------------------------------------------------

def load_attendance():

    if not ATTENDANCE_FILE.exists():
        return []

    with open(
        ATTENDANCE_FILE,
        "r",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


# --------------------------------------------------
# FACE MATCHING
# --------------------------------------------------

def compare_embedding(
    query_embedding,
    registered_embeddings
):

    query_embedding = query_embedding / max(
        np.linalg.norm(query_embedding),
        1e-12
    )

    scores = registered_embeddings @ query_embedding

    top_scores = np.sort(scores)[
        -min(TOP_K_MATCH_SCORES, len(scores)):
    ]

    return float(np.mean(top_scores))


def find_best_match(embedding):

    best_match = None
    best_score = -1.0

    for roll, student in students.items():

        score = compare_embedding(
            embedding,
            student["embeddings"]
        )

        if score > best_score:

            best_score = score

            best_match = {
                "name": student["name"],
                "roll_number": roll,
                "similarity": score
            }

    return best_match


# --------------------------------------------------
# STUDENT REGISTRATION
# --------------------------------------------------

def process_registration(frame):

    results = engine.process_frame(frame)

    with registration_lock:

        if not registration_state["active"]:
            return frame

        if len(results) != 1:

            registration_state["message"] = (
                "Position exactly one face in the camera"
            )

            return frame

        result = results[0]

        x, y, w, h = result["bbox"]

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        if registration_state["capture_requested"]:

            embedding = result["embedding"].astype(
                np.float32
            )

            embedding = embedding / max(
                np.linalg.norm(embedding),
                1e-12
            )

            registration_state["embeddings"].append(
                embedding
            )

            registration_state["capture_requested"] = False

            count = len(
                registration_state["embeddings"]
            )

            registration_state["message"] = (
                f"Captured sample {count}/"
                f"{REQUIRED_REGISTRATION_SAMPLES}"
            )

            if count >= REQUIRED_REGISTRATION_SAMPLES:

                name = registration_state["name"]
                roll = registration_state["roll_number"]

                embeddings = np.stack(
                    registration_state["embeddings"]
                )

                os.makedirs(
                    KNOWN_FACES_DIR,
                    exist_ok=True
                )

                file_path = os.path.join(
                    KNOWN_FACES_DIR,
                    f"{roll}.npz"
                )

                np.savez_compressed(
                    file_path,
                    name=np.array(name),
                    roll_number=np.array(roll),
                    embeddings=embeddings
                )

                students[roll] = {
                    "name": name,
                    "embeddings": embeddings.copy()
                }

                registration_state["active"] = False

                registration_state["message"] = (
                    f"Registration successful: {name}"
                )

                print(
                    f"\nStudent registered successfully!"
                    f"\nName: {name}"
                    f"\nRoll Number: {roll}"
                    f"\nSamples: {len(embeddings)}"
                )

        cv2.putText(
            frame,
            registration_state["message"],
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    return frame


# --------------------------------------------------
# LIVE FACE RECOGNITION
# --------------------------------------------------

def process_recognition(frame):

    global candidate_roll
    global consecutive_frames

    results = engine.process_frame(frame)

    if len(results) != 1:

        candidate_roll = None
        consecutive_frames = 0

        if len(results) == 0:
            message = "No face detected"
        else:
            message = "Multiple faces detected"

        live_status.update({
            "message": message,
            "name": None,
            "roll_number": None,
            "similarity": None
        })

        return frame

    result = results[0]

    x, y, w, h = result["bbox"]

    match = find_best_match(
        result["embedding"]
    )

    if match is None:
        return frame

    score = match["similarity"]

    if score >= SIMILARITY_THRESHOLD:

        roll = match["roll_number"]

        if candidate_roll == roll:
            consecutive_frames += 1
        else:
            candidate_roll = roll
            consecutive_frames = 1

        if roll not in attendance_processed:

            message = (
                f"Confirming {consecutive_frames}/"
                f"{REQUIRED_CONSECUTIVE_FRAMES}"
            )

            if consecutive_frames >= REQUIRED_CONSECUTIVE_FRAMES:

                response = attendance_manager.mark_attendance(
                    match["name"],
                    roll
                )

                attendance_processed.add(roll)

                if response["success"]:
                    message = "Attendance marked successfully"
                else:
                    message = "Attendance already marked today"

                print(
                    f"\n{message}"
                    f"\nName: {match['name']}"
                    f"\nRoll Number: {roll}"
                )

                # Require a fresh liveness challenge for the next attendance attempt.
                global liveness_verified, liveness_detector
                liveness_verified = False
                if liveness_detector is not None:
                    liveness_detector.reset()
                live_status["liveness_verified"] = False
                live_status["liveness_state"] = "BLINK"

        else:

            message = "Attendance already processed this session"

        label = f"{match['name']} ({roll})"
        color = (0, 255, 0)

        live_status.update({
            "message": message,
            "name": match["name"],
            "roll_number": roll,
            "similarity": round(score, 3)
        })

    else:

        candidate_roll = None
        consecutive_frames = 0

        label = "UNKNOWN"
        color = (0, 0, 255)

        live_status.update({
            "message": "Unknown person",
            "name": None,
            "roll_number": None,
            "similarity": round(score, 3)
        })

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

    return frame


# --------------------------------------------------
# WEBCAM STREAM
# --------------------------------------------------

def generate_frames():

    global camera, liveness_detector, liveness_verified

    initialize_recognition()
    liveness_verified = False
    if liveness_detector is not None:
        liveness_detector.reset()

    with camera_lock:

        if camera is not None:
            return

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            camera.release()
            camera = None

            live_status.update({
                "running": False,
                "message": "Unable to open webcam"
            })
            if liveness_detector is not None:
                liveness_detector.close()
                liveness_detector = None
            return

        live_status.update({
            "running": True,
            "message": "Camera started"
        })

    try:

        while True:

            current_camera = camera

            if current_camera is None:
                break

            success, frame = current_camera.read()

            if not success:
                break

            if registration_state["active"]:
                frame = process_registration(frame)
            elif not liveness_verified:
                # Liveness must pass before ResNet recognition can run.
                frame, liveness_result = liveness_detector.process_frame(frame)
                live_status.update({
                    "message": f"Liveness: {liveness_result['state']}",
                    "name": None,
                    "roll_number": None,
                    "similarity": None,
                    "liveness_state": liveness_result["state"],
                    "liveness_verified": liveness_result["liveness_detected"]
                })

                if liveness_result["liveness_detected"]:
                    liveness_verified = True
                    live_status["message"] = "Liveness verified. Recognizing face..."
            else:
                frame = process_recognition(frame)

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

    finally:

        with camera_lock:

            if camera is not None:
                camera.release()
                camera = None

        if liveness_detector is not None:
            liveness_detector.close()
            liveness_detector = None

        liveness_verified = False
        live_status.update({
            "running": False,
            "message": "Camera is stopped",
            "liveness_state": "NOT_STARTED",
            "liveness_verified": False
        })


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("dashboard.html")


# --------------------------------------------------
# VIDEO FEED
# --------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# LIVE STATUS API
# --------------------------------------------------

@app.route("/api/live_status")
def get_live_status():

    return jsonify(live_status)


# --------------------------------------------------
# ATTENDANCE API
# --------------------------------------------------

@app.route("/api/attendance")
def get_attendance():

    records = load_attendance()

    selected_date = request.args.get("date")
    search = request.args.get(
        "search", ""
    ).strip().lower()

    if selected_date:

        records = [
            r for r in records
            if r["date"] == selected_date
        ]

    if search:

        records = [
            r for r in records
            if search in r["name"].lower()
            or search in r["roll_number"].lower()
        ]

    return jsonify({
        "success": True,
        "total": len(records),
        "records": records
    })


# --------------------------------------------------
# STATISTICS API
# --------------------------------------------------

@app.route("/api/statistics")
def get_statistics():

    records = load_attendance()

    unique_students = {
        r["roll_number"] for r in records
    }

    dates = sorted({
        r["date"] for r in records
    })

    return jsonify({
        "success": True,
        "total_entries": len(records),
        "unique_students": len(unique_students),
        "first_date": dates[0] if dates else None,
        "latest_date": dates[-1] if dates else None
    })


# --------------------------------------------------
# CSV EXPORT API
# --------------------------------------------------

@app.route("/api/attendance/export")
def export_attendance():

    records = load_attendance()

    selected_date = request.args.get("date")

    if selected_date:

        records = [
            r for r in records
            if r["date"] == selected_date
        ]

    output = io.StringIO()

    fields = [
        "name",
        "roll_number",
        "date",
        "time"
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(records)

    return send_file(
        io.BytesIO(
            output.getvalue().encode("utf-8")
        ),
        mimetype="text/csv",
        as_attachment=True,
        download_name="attendance_export.csv"
    )


# --------------------------------------------------
# START STUDENT REGISTRATION
# --------------------------------------------------

@app.route("/api/register/start", methods=["POST"])
def start_registration():

    initialize_recognition()

    data = request.get_json(silent=True) or {}

    name = data.get("name", "").strip()
    roll = data.get("roll_number", "").strip()

    if not name or not roll:

        return jsonify({
            "success": False,
            "message": "Name and roll number are required"
        }), 400

    if not re.fullmatch(r"[A-Za-z0-9_-]+", roll):

        return jsonify({
            "success": False,
            "message": "Invalid roll number"
        }), 400

    file_path = os.path.join(
        KNOWN_FACES_DIR,
        f"{roll}.npz"
    )

    if roll in students or os.path.exists(file_path):

        return jsonify({
            "success": False,
            "message": "Student is already registered"
        }), 409

    if camera is None or not camera.isOpened():

        return jsonify({
            "success": False,
            "message": "Start the webcam before registration"
        }), 400

    with registration_lock:

        if registration_state["active"]:

            return jsonify({
                "success": False,
                "message": "Another registration is already active"
            }), 409

        registration_state.update({
            "active": True,
            "name": name,
            "roll_number": roll,
            "embeddings": [],
            "capture_requested": False,
            "message": "Position your face and capture samples"
        })

    return jsonify({
        "success": True,
        "message": "Registration session started"
    })


# --------------------------------------------------
# CAPTURE REGISTRATION SAMPLE
# --------------------------------------------------

@app.route("/api/register/capture", methods=["POST"])
def capture_registration_sample():

    with registration_lock:

        if not registration_state["active"]:

            return jsonify({
                "success": False,
                "message": "No active registration"
            }), 400

        if registration_state["capture_requested"]:

            return jsonify({
                "success": False,
                "message": "Previous capture is still processing"
            }), 409

        if len(registration_state["embeddings"]) >= REQUIRED_REGISTRATION_SAMPLES:

            return jsonify({
                "success": False,
                "message": "All samples already captured"
            }), 400

        registration_state["capture_requested"] = True

    return jsonify({
        "success": True,
        "message": "Capture requested"
    })


# --------------------------------------------------
# REGISTRATION STATUS
# --------------------------------------------------

@app.route("/api/register/status")
def registration_status():

    with registration_lock:

        return jsonify({
            "active": registration_state["active"],
            "captured": len(
                registration_state["embeddings"]
            ),
            "required": REQUIRED_REGISTRATION_SAMPLES,
            "message": registration_state["message"]
        })


# --------------------------------------------------
# CANCEL REGISTRATION
# --------------------------------------------------

@app.route("/api/register/cancel", methods=["POST"])
def cancel_registration():

    with registration_lock:

        registration_state.update({
            "active": False,
            "name": "",
            "roll_number": "",
            "embeddings": [],
            "capture_requested": False,
            "message": "Registration cancelled"
        })

    return jsonify({
        "success": True,
        "message": "Registration cancelled"
    })


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=SERVER_DEBUG,
        threaded=True
    )
