/* =========================================================
   CAMPUSGUARD MAIN SCRIPT (PRODUCTION READY)
   ========================================================= */

const BASE_URL = "https://campus-guard-light.onrender.com";

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
    const response = await fetch(`${BASE_URL}/lost-found/analyze`, {
      method: "POST",
      body: formData
    });

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


/* ---------------- VIOLENCE ---------------- */
function detectViolence() {
  const fileInput = document.getElementById("violenceVideo");
  const result = document.getElementById("violenceResult");

  if (!fileInput.files.length) {
    result.innerHTML = "⚠️ Please upload a video";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  result.innerHTML = "⏳ <b>Analyzing video...</b>";

  fetch(`${BASE_URL}/violence/predict`, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {

      if (data.error) {
        result.innerHTML = `❌ <b>Error:</b> ${data.error}`;
        return;
      }

      if (data.result === "Violence Detected") {
        result.innerHTML = `
🚨 <b style="color:red;">VIOLENCE DETECTED</b><br>
Confidence: ${data.confidence}`;
      } else {
        result.innerHTML = `
✅ <b style="color:green;">NO VIOLENCE</b><br>
Confidence: ${data.confidence}`;
      }

    })
    .catch(() => {
      result.innerHTML = "❌ Backend not reachable";
    });
}


/* ---------------- KEYWORD DETECTION ---------------- */

let mediaRecorder;
let audioChunks = [];

function startRecording() {
  audioChunks = [];
  document.getElementById("emergencyStatus").innerText = "Recording...";

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    });
}

function stopRecording() {
  mediaRecorder.stop();

  mediaRecorder.onstop = () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    sendEmergencyAudio(audioBlob);
  };
}

function detectEmergency() {
  const fileInput = document.getElementById("emergencyAudio");

  if (!fileInput.files.length) {
    alert("Please select audio");
    return;
  }

  sendEmergencyAudio(fileInput.files[0]);
}

function sendEmergencyAudio(audioBlob) {
  const status = document.getElementById("emergencyStatus");
  const result = document.getElementById("emergencyResult");

  status.innerText = "Processing...";
  result.innerText = "";

  const formData = new FormData();
  formData.append("file", audioBlob);

  fetch(`${BASE_URL}/keyword/predict-audio`, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      status.innerText = "Done";

      result.innerHTML =
        `🗣 ${data.recognized_text || "N/A"} <br>
🚨 ${data.prediction}`;
    })
    .catch(() => {
      status.innerText = "Error";
      result.innerText = "❌ Backend not reachable";
    });
}


/* ---------------- ABUSE DETECTION ---------------- */

let abuseRecorder;
let abuseChunks = [];

function startAbuseRecording() {
  abuseChunks = [];
  document.getElementById("abuseStatus").innerText = "Recording...";

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      abuseRecorder = new MediaRecorder(stream);
      abuseRecorder.start();
      abuseRecorder.ondataavailable = e => abuseChunks.push(e.data);
    });
}

function stopAbuseRecording() {
  abuseRecorder.stop();

  abuseRecorder.onstop = () => {
    const audioBlob = new Blob(abuseChunks, { type: "audio/webm" });
    sendAbuseAudio(audioBlob);
  };
}

function detectAbuse() {
  const fileInput = document.getElementById("abuseAudio");

  if (!fileInput.files.length) {
    alert("Select audio");
    return;
  }

  sendAbuseAudio(fileInput.files[0]);
}

function sendAbuseAudio(audioBlob) {
  const status = document.getElementById("abuseStatus");
  const result = document.getElementById("abuseResult");

  status.innerText = "Processing...";
  result.innerText = "";

  const formData = new FormData();
  formData.append("file", audioBlob);

  fetch(`${BASE_URL}/abuse/predict-audio`, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      status.innerText = "Done";

      result.innerHTML =
        `🗣 ${data.recognized_text || "N/A"} <br>
⚠️ ${data.result}`;
    })
    .catch(() => {
      status.innerText = "Error";
      result.innerText = "❌ Backend not reachable";
    });
}


/* ---------------- SCROLL ANIMATION ---------------- */
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
