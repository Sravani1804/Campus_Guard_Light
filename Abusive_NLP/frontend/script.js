function uploadAudio() {
    console.log("FUNCTION CALLED");

    let fileInput = document.getElementById("audioInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please upload an audio file!");
        return;
    }

    let formData = new FormData();
    formData.append("audio", file);

    // Show loading
    document.getElementById("loading").style.display = "block";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())   // 🔥 IMPORTANT FIX
    .then(data => {
        console.log("API DATA:", data);

        // Hide loading
        document.getElementById("loading").style.display = "none";

        // Show transcription
        document.getElementById("text").innerText =
            data.transcription || "No text detected";

        let pred = document.getElementById("prediction");

        // Handle probability safely
        let prob = data.probability ?? 0;
        let confidence = (prob * 100).toFixed(2);

        // Show prediction with color + emoji
        if (data.prediction === "Abusive") {
            pred.innerText = "🚨 Abusive Speech Detected";
            pred.className = "abusive";
        } else {
            pred.innerText = "✅ Safe Speech";
            pred.className = "safe";
        }

        // Show confidence
        document.getElementById("confidenceBox").innerText =
            "Confidence: " + confidence + "%";
    })
    .catch(error => {
        console.error("Error:", error);

        document.getElementById("loading").style.display = "none";

        document.getElementById("text").innerText = "Error processing audio";
        document.getElementById("prediction").innerText = "Error";
        document.getElementById("confidenceBox").innerText = "";
    });
}