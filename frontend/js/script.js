/* =========================================================
   CAMPUSGUARD MAIN SCRIPT (CLEAN + UPDATED)
   ========================================================= */

/* ---------------- LOST & FOUND ---------------- */
async function findLostItem(event, btn) {
  event.preventDefault();

  const card = btn.closest(".demo-card");
  const resultSpan = card.querySelector(".result span");

  const image = document.getElementById("lostImage").files[0];
  const video = document.getElementById("lostVideo").files[0];

  if (!image || !video) {
    resultSpan.innerText = "⚠️ Please upload both image and video";
    return;
  }

  const formData = new FormData();
  formData.append("lost_image", image);
  formData.append("video", video);

  resultSpan.innerText = "⏳ Processing request...";
  btn.disabled = true;

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/lost-found/analyze",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json();

    if (data.status === "MATCH_FOUND") {
      resultSpan.innerText =
`✅ MATCH FOUND
Camera ID : ${data.camera_id}
Room No   : ${data.room_no}
Confidence: ${data.confidence}
Timestamp : ${data.timestamp} seconds`;
    } else {
      resultSpan.innerHTML = `
❌ NO MATCH FOUND <br>
The object was not detected in the given CCTV footage.`;
    }

  } catch (err) {
    resultSpan.innerText = "❌ Backend not reachable";
  }

  btn.disabled = false;
}


/* ---------------- VIOLENCE & ABUSE (PLACEHOLDER) ---------------- */
function analyzeViolence(btn) {
  showPendingStatus(btn, "Backend AI model not connected yet");
}

function analyzeAbuseAudio(btn) {
  showPendingStatus(btn, "Audio-based abuse detection model pending");
}

function showPendingStatus(btn, message) {
  const card = btn.closest(".demo-card");
  const result = card.querySelector(".result span");
  result.innerText = message;
}


/* =========================================================
   🚨 KEYWORD DETECTION (UPLOAD + LIVE RECORDING)
   ========================================================= */

let mediaRecorder;
let audioChunks = [];

/* -------- START RECORDING -------- */
function startRecording() {
  audioChunks = [];

  const status = document.getElementById("emergencyStatus");
  status.innerText = "Status: Recording...";

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();

      mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
      };
    })
    .catch(() => {
      status.innerText = "Status: Mic permission denied";
    });
}


/* -------- STOP RECORDING -------- */
function stopRecording() {
  if (!mediaRecorder) return;

  const status = document.getElementById("emergencyStatus");
  status.innerText = "Status: Processing...";

  mediaRecorder.stop();

  mediaRecorder.onstop = () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    sendEmergencyAudio(audioBlob);
  };
}


/* -------- FILE UPLOAD -------- */
function detectEmergency() {
  const fileInput = document.getElementById("emergencyAudio");

  if (!fileInput.files.length) {
    alert("Please select an audio file");
    return;
  }

  sendEmergencyAudio(fileInput.files[0]);
}


/* -------- COMMON BACKEND CALL -------- */
function sendEmergencyAudio(audioBlob) {
  const status = document.getElementById("emergencyStatus");
  const result = document.getElementById("emergencyResult");

  status.innerText = "Status: Processing...";
  result.innerText = "";

  const formData = new FormData();
  formData.append("file", audioBlob);

  fetch("http://127.0.0.1:8000/keyword/predict-audio", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      status.innerText = "Status: Done";

      const text = data.recognized_text || data.text || "N/A";
      const prediction = data.prediction || data.result || "N/A";

      result.innerHTML =
        `🗣 <b>Text:</b> ${text}<br>🚨 <b>Result:</b> ${prediction}`;
    })
    .catch(() => {
      status.innerText = "Status: Error";
      result.innerText = "❌ Backend not reachable";
    });
}

/* =========================================================
   🗣 ABUSIVE DETECTION (UPLOAD + LIVE RECORDING)
   ========================================================= */

let abuseRecorder;
let abuseChunks = [];

/* -------- START RECORDING -------- */
function startAbuseRecording() {
  abuseChunks = [];

  const status = document.getElementById("abuseStatus");
  status.innerText = "Status: Recording...";

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      abuseRecorder = new MediaRecorder(stream);
      abuseRecorder.start();

      abuseRecorder.ondataavailable = e => {
        abuseChunks.push(e.data);
      };
    })
    .catch(() => {
      status.innerText = "Status: Mic permission denied";
    });
}


/* -------- STOP RECORDING -------- */
function stopAbuseRecording() {
  if (!abuseRecorder) return;

  const status = document.getElementById("abuseStatus");
  status.innerText = "Status: Processing...";

  abuseRecorder.stop();

  abuseRecorder.onstop = () => {
    const audioBlob = new Blob(abuseChunks, { type: "audio/webm" });
    sendAbuseAudio(audioBlob);
  };
}


/* -------- FILE UPLOAD -------- */
function detectAbuse() {
  const fileInput = document.getElementById("abuseAudio");

  if (!fileInput.files.length) {
    alert("Please select an audio file");
    return;
  }

  sendAbuseAudio(fileInput.files[0]);
}


/* -------- COMMON BACKEND CALL -------- */
function sendAbuseAudio(audioBlob) {
  const status = document.getElementById("abuseStatus");
  const result = document.getElementById("abuseResult");

  status.innerText = "Status: Processing...";
  result.innerText = "";

  const formData = new FormData();
  formData.append("file", audioBlob);

  fetch("http://127.0.0.1:8000/abuse/predict-audio", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      status.innerText = "Status: Done";

      console.log("ABUSE API:", data);

      const text = data.recognized_text || "N/A";
      const prediction = data.result || "N/A";

      result.innerHTML =
        `🗣 <b>Text:</b> ${text}<br>⚠️ <b>Result:</b> ${prediction}`;
    })
    .catch(() => {
      status.innerText = "Status: Error";
      result.innerText = "❌ Backend not reachable";
    });
}

/* =========================================================
   SCROLL ANIMATION
   ========================================================= */
const reveals = document.querySelectorAll(".reveal");

function revealOnScroll() {
  reveals.forEach(el => {
    if (el.getBoundingClientRect().top < window.innerHeight - 100) {
      el.classList.add("active");
    }
  });
}

window.addEventListener("scroll", revealOnScroll);
window.addEventListener("load", revealOnScroll);