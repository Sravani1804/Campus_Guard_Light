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
        The object was not detected in the given CCTV footage.
      `;
    }

  } catch (err) {
    resultSpan.innerText = "❌ Backend not reachable";
  }

  btn.disabled = false;
}



function analyzeViolence(btn) {
  showPendingStatus(btn, "Backend AI model not connected yet");
}

function analyzeAbuseAudio(btn) {
  showPendingStatus(btn, "Audio-based abuse detection model pending");
}

async function detectKeywordAudio(btn) {
  const card = btn.closest(".demo-card");
  const resultSpan = card.querySelector(".result span");
  const audioFile = card.querySelector("input[type='file']").files[0];

  if (!audioFile) {
    resultSpan.innerText = "⚠️ Please upload an audio file";
    return;
  }

  const formData = new FormData();
  formData.append("audio", audioFile);

  resultSpan.innerText = "⏳ Processing...";
  btn.disabled = true;

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/keyword/predict-audio",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json();

    resultSpan.innerText =
      `Text: ${data.text} | Result: ${data.prediction}`;

  } catch (err) {
    resultSpan.innerText = "❌ Backend not reachable";
  }

  btn.disabled = false;
}


/* Scroll reveal animation */
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
function detectEmergency() {
  const fileInput = document.getElementById("emergencyAudio");
  const status = document.getElementById("emergencyStatus");
  const result = document.getElementById("emergencyResult");

  if (!fileInput || fileInput.files.length === 0) {
    alert("Please select an audio file");
    return;
  }

  status.innerText = "Status: Processing...";
  result.innerText = "";

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  fetch("http://127.0.0.1:8000/keyword/predict-audio", {
    method: "POST",
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      status.innerText = "Status: Done";
      result.innerHTML =
        "🗣 Text: " + data.recognized_text +
        "<br>🚨 Result: " + data.prediction;
    })
    .catch(error => {
      console.error(error);
      status.innerText = "Status: Error";
      result.innerText = "Backend not reachable";
    });
}
