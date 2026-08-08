function dataURLtoBlob(dataUrl) {
    const arr = dataUrl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) { u8arr[n] = bstr.charCodeAt(n); }
    return new Blob([u8arr], { type: mime });
}

// Helper function to dynamically update the dashboard bars
function updateDashboard(rgbScore, noiseScore) {
    const rgbPct = Math.round(rgbScore * 100);
    const noisePct = Math.round(noiseScore * 100);

    document.getElementById('rgbVal').innerText = `${rgbPct}%`;
    document.getElementById('noiseVal').innerText = `${noisePct}%`;

    document.getElementById('rgbBar').style.width = `${rgbPct}%`;
    document.getElementById('noiseBar').style.width = `${noisePct}%`;

    // Strict threshold: turns red if risk is > 55%
    document.getElementById('rgbBar').style.backgroundColor = rgbPct > 55 ? '#dc3545' : '#28a745';
    document.getElementById('noiseBar').style.backgroundColor = noisePct > 55 ? '#dc3545' : '#28a745';
}

document.getElementById('scanBtn').addEventListener('click', async () => {
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = "Taking secure screenshot...";
    updateDashboard(0, 0);

    try {
        // THE FIX: Use Chrome's native screenshot API!
        // This captures whatever is visible on the screen and bypasses ALL website security blockers.
        chrome.tabs.captureVisibleTab(null, { format: "jpeg", quality: 100 }, async (dataUrl) => {
            
            if (chrome.runtime.lastError || !dataUrl) {
                statusDiv.innerText = "Capture failed. Make sure you are on a valid webpage.";
                return;
            }

            statusDiv.innerText = "Processing image...";
            const imageBlob = dataURLtoBlob(dataUrl);
            const formData = new FormData();
            formData.append('file', imageBlob, 'frame.jpg');

            statusDiv.innerText = "Querying ML Model...";

            try {
                // Using 127.0.0.1 is much safer for Chrome Extensions than localhost
                const backendUrl = 'http://127.0.0.1:8000/predict';
                const apiResponse = await fetch(backendUrl, { method: 'POST', body: formData });

                if (!apiResponse.ok) throw new Error(`API Error: ${apiResponse.status}`);

                const result = await apiResponse.json();

                if (result.error) throw new Error(result.error);

                updateDashboard(result.rgb_score, result.noise_score);

                if (result.decision === "FAKE") {
                    statusDiv.innerHTML = `<span style="color: #dc3545;">🚨 DEEPFAKE DETECTED (${result.confidence_score})</span>`;
                } else if (result.decision === "REAL") {
                    statusDiv.innerHTML = `<span style="color: #28a745;">✅ VERIFIED REAL (${result.confidence_score})</span>`;
                } else {
                    statusDiv.innerHTML = `<span style="color: #f59e0b;">⚠️ UNCERTAIN (${result.confidence_score})</span>`;
                }

            } catch (err) {
                console.error("Backend Connection Error:", err);
                // REMOVED THE SIMULATION. NOW SHOWS THE ACTUAL ERROR!
                statusDiv.innerHTML = `<span style="color: #dc3545;">❌ Backend Offline or Failed. Is FastAPI running?</span>`;
                updateDashboard(0, 0);
            }
        });

    } catch (error) {
        console.error(error);
        statusDiv.innerText = "An error occurred.";
    }
});