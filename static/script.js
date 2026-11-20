
const API = "/api";

let cameraActive = false;
let statusTimer = null;
let registrationTimer = null;
let registrationActive = false;

const byId = (id) => document.getElementById(id);

async function requestJSON(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.message || `Request failed (${response.status})`);
    }

    return data;
}

async function loadStatistics() {
    try {
        const data = await requestJSON(`${API}/statistics`);

        byId("totalEntries").textContent = data.total_entries ?? 0;
        byId("uniqueStudents").textContent = data.unique_students ?? 0;
        byId("latestDate").textContent = data.latest_date || "--";

    } catch (error) {
        console.error("Statistics error:", error);
    }
}

async function loadAttendance() {
    const params = new URLSearchParams();

    const search = byId("searchInput").value.trim();
    const date = byId("dateFilter").value;

    if (search) params.set("search", search);
    if (date) params.set("date", date);

    try {
        const data = await requestJSON(`${API}/attendance?${params}`);
        const tbody = byId("attendanceBody");

        tbody.replaceChildren();

        if (!data.records?.length) {
            const row = tbody.insertRow();
            const cell = row.insertCell();

            cell.colSpan = 5;
            cell.textContent = "No attendance records found.";

            return;
        }

        data.records.forEach(record => {
            const row = tbody.insertRow();

            [
                record.name,
                record.roll_number,
                record.date,
                record.time,
                "Present"
            ].forEach(value => {
                const cell = row.insertCell();
                cell.textContent = value ?? "--";
            });
        });

    } catch (error) {
        console.error("Attendance error:", error);
    }
}

function setPipeline(detailId, badgeId, detail, label, state = "pending") {
    byId(detailId).textContent = detail;

    const badge = byId(badgeId);

    badge.textContent = label;
    badge.className = `verification-badge ${state}`;
}

function updatePipeline(data) {
    const stage = String(data.liveness_state || "").toUpperCase();

    const verified =
        Boolean(data.liveness_verified) || stage === "PASSED";

    let liveText = data.message || "Waiting for camera";

    if (stage === "BLINK") {
        liveText = "Please blink naturally";
    } else if (stage === "CALIBRATING") {
        liveText = "Calibrating neutral head position";
    } else if (stage === "TURN") {
        liveText = "Turn your head in the prompted direction";
    } else if (stage === "RETURN") {
        liveText = "Return your head to the center";
    } else if (stage === "PASSED") {
        liveText = "Liveness verification passed";
    } else if (stage === "FAILED") {
        liveText = data.message || "Liveness verification failed";
    }

    setPipeline(
        "livenessDetail",
        "livenessBadge",
        liveText,
        verified
            ? "Verified"
            : stage === "FAILED"
                ? "Failed"
                : data.running
                    ? "Checking"
                    : "Waiting",
        verified
            ? "success"
            : stage === "FAILED"
                ? "failed"
                : data.running
                    ? "progress"
                    : "pending"
    );

    const hasIdentity = Boolean(data.name && data.roll_number);

    setPipeline(
        "identityDetail",
        "identityBadge",
        hasIdentity
            ? `${data.name} (${data.roll_number}) · Similarity ${Number(data.similarity || 0).toFixed(3)}`
            : verified
                ? "Liveness passed; waiting for identity match"
                : "Identity check begins after liveness",
        hasIdentity
            ? "Matched"
            : verified
                ? "Recognizing"
                : "Waiting",
        hasIdentity
            ? "success"
            : verified
                ? "progress"
                : "pending"
    );

    const message = String(data.message || "").toLowerCase();

    const marked =
        message.includes("attendance marked") ||
        message.includes("attendance recorded");

    const duplicate =
        message.includes("already marked") ||
        message.includes("already processed");

    setPipeline(
        "attendanceDetail",
        "attendanceBadge",
        marked
            ? "Attendance successfully recorded"
            : duplicate
                ? "Attendance already processed this session"
                : hasIdentity
                    ? "Identity matched; processing attendance"
                    : "Waiting for identity match",
        marked
            ? "Marked"
            : duplicate
                ? "Already recorded"
                : "Waiting",
        marked || duplicate
            ? "success"
            : "pending"
    );
}

async function updateLiveStatus() {
    try {
        const data = await requestJSON(`${API}/live_status`);

        byId("recognitionStatus").textContent =
            data.message || "Waiting for camera";

        byId("studentName").textContent = data.name || "--";
        byId("studentRoll").textContent = data.roll_number || "--";

        byId("similarity").textContent =
            data.similarity == null
                ? "--"
                : Number(data.similarity).toFixed(3);

        const badge = byId("cameraBadge");

        badge.textContent = data.running
            ? "Camera running"
            : "Camera stopped";

        badge.classList.toggle("active", Boolean(data.running));

        updatePipeline(data);

        if (
            data.message &&
            /attendance marked|already processed|already marked/i.test(data.message)
        ) {
            loadAttendance();
            loadStatistics();
        }

    } catch (error) {
        console.error("Live status error:", error);
    }
}

// Keep the original Flask MJPEG image stream.
// Do not wait for img.onload because MJPEG streams are long-lived.

function startCamera() {
    if (cameraActive) return;

    const feed = byId("webcamFeed");
    const placeholder = byId("cameraPlaceholder");

    feed.src = `/video_feed?t=${Date.now()}`;

    feed.style.display = "block";
    placeholder.style.display = "none";

    cameraActive = true;

    byId("startCamera").disabled = true;
    byId("stopCamera").disabled = false;

    if (statusTimer) {
        clearInterval(statusTimer);
    }

    statusTimer = setInterval(() => {
        updateLiveStatus();
        loadAttendance();
        loadStatistics();
    }, 1500);

    updateLiveStatus();
}

function stopCamera() {
    const feed = byId("webcamFeed");

    feed.removeAttribute("src");
    feed.style.display = "none";

    byId("cameraPlaceholder").style.display = "flex";

    cameraActive = false;

    byId("startCamera").disabled = false;
    byId("stopCamera").disabled = true;

    if (statusTimer) {
        clearInterval(statusTimer);
    }

    statusTimer = null;

    updateLiveStatus();
}

function setRegistrationControls(active) {
    registrationActive = active;

    byId("beginRegistration").disabled = active;
    byId("captureSample").disabled = !active;
    byId("cancelRegistration").disabled = !active;

    const quickBtn = byId("quickCaptureSample");
    if (quickBtn) {
        quickBtn.style.display = active ? "inline-flex" : "none";
        quickBtn.disabled = !active;
    }

    byId("studentNameInput").disabled = active;
    byId("studentRollInput").disabled = active;
}

function updateRegistrationProgress(data) {
    const captured = Number(data.captured || 0);
    const required = Number(data.required || 10);

    byId("sampleCount").textContent = captured;

    const quickCounter = byId("quickSampleCounter");
    if (quickCounter) {
        quickCounter.textContent = `${captured}/${required}`;
    }

    byId("progressBar").style.width =
        `${Math.min(captured / required * 100, 100)}%`;

    byId("registrationMessage").textContent =
        data.message || "Registration in progress";

    if (!data.active && registrationActive) {
        setRegistrationControls(false);

        if (registrationTimer) {
            clearInterval(registrationTimer);
        }

        registrationTimer = null;

        loadStatistics();
    }
}

async function beginRegistration() {
    const name = byId("studentNameInput").value.trim();
    const roll = byId("studentRollInput").value.trim();

    if (!name || !roll) {
        return alert("Enter both student name and roll number.");
    }

    if (!cameraActive) {
        return alert("Start the webcam before registering a student.");
    }

    try {
        const data = await requestJSON(`${API}/register/start`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                name,
                roll_number: roll
            })
        });

        setRegistrationControls(true);

        byId("sampleCount").textContent = "0";
        byId("progressBar").style.width = "0%";

        byId("registrationMessage").textContent =
            data.message || "Registration started";

        if (registrationTimer) {
            clearInterval(registrationTimer);
        }

        registrationTimer = setInterval(
            updateRegistrationStatus,
            500
        );

    } catch (error) {
        alert(error.message || "Unable to start registration.");
    }
}

async function captureSample() {
    const button = byId("captureSample");

    button.disabled = true;

    try {
        const data = await requestJSON(
            `${API}/register/capture`,
            { method: "POST" }
        );

        byId("registrationMessage").textContent =
            data.message ||
            "Capture requested. Keep your face visible.";

    } catch (error) {
        alert(error.message);

    } finally {
        if (registrationActive) {
            button.disabled = false;
        }
    }
}

async function updateRegistrationStatus() {
    try {
        const data = await requestJSON(`${API}/register/status`);

        updateRegistrationProgress(data);

    } catch (error) {
        console.error("Registration status error:", error);
    }
}

async function cancelRegistration() {
    try {
        const data = await requestJSON(
            `${API}/register/cancel`,
            { method: "POST" }
        );

        byId("registrationMessage").textContent =
            data.message || "Registration cancelled";

        setRegistrationControls(false);

        if (registrationTimer) {
            clearInterval(registrationTimer);
        }

        registrationTimer = null;

    } catch (error) {
        alert(error.message);
    }
}

function exportAttendance() {
    const params = new URLSearchParams();

    if (byId("dateFilter").value) {
        params.set("date", byId("dateFilter").value);
    }

    window.location.href =
        `${API}/attendance/export?${params}`;
}

// Camera controls
byId("startCamera").addEventListener(
    "click",
    startCamera
);

byId("stopCamera").addEventListener(
    "click",
    stopCamera
);

// Quick Capture button under camera feed
const quickCaptureBtn = byId("quickCaptureSample");
if (quickCaptureBtn) {
    quickCaptureBtn.addEventListener("click", captureSample);
}

// Spacebar shortcut for instant sample capture
window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && registrationActive) {
        const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
        if (tag !== "input" && tag !== "textarea") {
            e.preventDefault();
            const btn = byId("captureSample");
            if (btn && !btn.disabled) {
                captureSample();
            }
        }
    }
});

// Tab Switcher between Live Recognition and Registration
const tabRecognition = byId("tabRecognition");
const tabRegistration = byId("tabRegistration");
const panelRecognitionContent = byId("panelRecognitionContent");
const panelRegistrationContent = byId("panelRegistrationContent");

function switchStageTab(tab) {
    if (tab === "registration") {
        if (tabRegistration) tabRegistration.classList.add("active");
        if (tabRecognition) tabRecognition.classList.remove("active");
        if (panelRegistrationContent) panelRegistrationContent.style.display = "block";
        if (panelRecognitionContent) panelRecognitionContent.style.display = "none";
    } else {
        if (tabRecognition) tabRecognition.classList.add("active");
        if (tabRegistration) tabRegistration.classList.remove("active");
        if (panelRecognitionContent) panelRecognitionContent.style.display = "block";
        if (panelRegistrationContent) panelRegistrationContent.style.display = "none";
    }
}

if (tabRecognition) {
    tabRecognition.addEventListener("click", () => switchStageTab("recognition"));
}

if (tabRegistration) {
    tabRegistration.addEventListener("click", () => switchStageTab("registration"));
}

// Connect sidebar links to tab switching
const regNavLink = document.querySelector('.sidebar-nav a[href="#studentRegistration"]');
if (regNavLink) {
    regNavLink.addEventListener("click", (e) => {
        switchStageTab("registration");
    });
}

const liveNavLink = document.querySelector('.sidebar-nav a[href="#liveRecognition"]');
if (liveNavLink) {
    liveNavLink.addEventListener("click", (e) => {
        switchStageTab("recognition");
    });
}

// Registration controls
byId("beginRegistration").addEventListener(
    "click",
    beginRegistration
);

byId("captureSample").addEventListener(
    "click",
    captureSample
);

byId("cancelRegistration").addEventListener(
    "click",
    cancelRegistration
);

// Attendance controls
byId("searchInput").addEventListener(
    "input",
    loadAttendance
);

byId("dateFilter").addEventListener(
    "change",
    loadAttendance
);

byId("resetButton").addEventListener("click", () => {
    byId("searchInput").value = "";
    byId("dateFilter").value = "";

    loadAttendance();
});

byId("exportButton").addEventListener(
    "click",
    exportAttendance
);

byId("stopCamera").disabled = true;

// Initial dashboard loading
loadStatistics();
loadAttendance();
updateLiveStatus();
