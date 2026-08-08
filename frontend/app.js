// --- SPA Page Router Controller ---
const navItems = document.querySelectorAll('.nav-item');
const pageViews = document.querySelectorAll('.page-view');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        navItems.forEach(nav => nav.classList.remove('active'));
        pageViews.forEach(view => view.classList.remove('active'));
        item.classList.add('active');
        document.getElementById(item.getAttribute('data-target')).classList.add('active');
    });
});

// --- UI Elements & Setup ---
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const videoPlayer = document.getElementById('videoPlayer');
const analyzeBtn = document.getElementById('analyzeBtn');
const sidebarFilename = document.getElementById('sidebarFilename');
const verdictDisplay = document.getElementById('verdictDisplay');
let videoFile = null;

uploadBox.addEventListener('click', () => fileInput.click());
uploadBox.addEventListener('dragover', (e) => e.preventDefault());
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) processMedia(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) processMedia(e.target.files[0]);
});

function processMedia(file) {
    videoFile = file;
    uploadBox.style.display = 'none';
    videoPlayer.style.display = 'block';
    videoPlayer.src = URL.createObjectURL(file);
    sidebarFilename.innerText = file.name;
    analyzeBtn.disabled = false;
    verdictDisplay.innerHTML = `<span style="color:var(--text-main); font-weight:600;">Media Loaded</span><br>Ready for deep analysis.`;
    resetPresentationMetrics();
}

// --- Animation Helpers ---
function setCircularGauge(elementId, textId, score) {
    const circle = document.getElementById(elementId);
    const text = document.getElementById(textId);
    if (!circle || !text) return;
    const radius = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = circumference - (score * circumference);
    text.innerText = `${Math.round(score * 100)}%`;
}

function resetPresentationMetrics() {
    setCircularGauge('rgbGauge', 'rgbText', 0);
    setCircularGauge('noiseGauge', 'noiseText', 0);
    if(document.getElementById('sparklinePath1')) document.getElementById('sparklinePath1').setAttribute('d', 'M 0 60');
    if(document.getElementById('sparklinePath2')) document.getElementById('sparklinePath2').setAttribute('d', 'M 0 60');
    for(let i=0; i<5; i++) {
        const cell = document.getElementById(`cell-${i}`);
        if(cell) {
            cell.style.backgroundColor = '#374151';
            cell.innerText = '-';
            cell.style.color = 'var(--text-main)';
        }
    }
}

// --- Main Analysis Engine ---
analyzeBtn.addEventListener('click', async () => {
    analyzeBtn.disabled = true;
    verdictDisplay.innerText = "Transmitting to OpenCV Backend... Please wait.";
    resetPresentationMetrics();

    const formData = new FormData();
    formData.append("file", videoFile);

    try {
        // 1. Get real data from Python
        const res = await fetch('/analyze-video', { method: 'POST', body: formData });
        if (!res.ok) throw new Error("Backend connection failed.");
        const data = await res.json();
        
        // Ensure we have an array (fallback to zeros if missing)
        const scores = data.frame_scores || [0, 0, 0, 0, 0];
        
        let sumRisk = 0;
        let pathCoord1 = "M 0 50";
        let pathCoord2 = "M 0 50";

        // 2. Play the animation using real backend frame data
        verdictDisplay.innerText = "Parsing Neural Streams...";
        for (let i = 0; i < scores.length; i++) {
            await new Promise(r => setTimeout(r, 350)); // Visual delay effect
            
            let riskRatio = scores[i] / 100; // Convert to 0.0 - 1.0 format
            sumRisk += riskRatio;
            
            // Update Gauges
            setCircularGauge('rgbGauge', 'rgbText', sumRisk / (i+1));
            setCircularGauge('noiseGauge', 'noiseText', (sumRisk / (i+1)) * 0.95); // Slight variance for visuals
            
            // Update Sparklines
            const xCoord = (370 / (scores.length - 1)) * i;
            pathCoord1 += ` L ${xCoord} ${60 - (riskRatio * 50)}`;
            pathCoord2 += ` L ${xCoord} ${60 - (riskRatio * 45)}`; 
            if(document.getElementById('sparklinePath1')) document.getElementById('sparklinePath1').setAttribute('d', pathCoord1);
            if(document.getElementById('sparklinePath2')) document.getElementById('sparklinePath2').setAttribute('d', pathCoord2);
            
            // Update Timeline Cells
            const cell = document.getElementById(`cell-${i}`);
            if(cell) {
                cell.innerText = `${Math.round(riskRatio * 100)}%`;
                if (riskRatio > 0.65) {
                    cell.style.backgroundColor = 'var(--color-red)';
                    cell.style.color = '#ffffff';
                } else if (riskRatio > 0.45) {
                    cell.style.backgroundColor = 'var(--color-orange)';
                    cell.style.color = '#ffffff';
                } else {
                    cell.style.backgroundColor = 'var(--color-green)';
                    cell.style.color = '#ffffff';
                }
            }
        }

        // 3. Final Verdict Banner
        if (data.decision === "FAKE") {
            verdictDisplay.innerHTML = `<span style="color:var(--color-red); font-weight:700; font-size:14px;">🚨 CORES COMPROMISED</span><br>Synthetic modifications detected. Confidence: ${data.confidence_score}`;
        } else if (data.decision === "REAL") {
            verdictDisplay.innerHTML = `<span style="color:var(--color-green); font-weight:700; font-size:14px;">✅ AUTHENTIC DATA BLOCK</span><br>Video profile matches baseline. Confidence: ${data.confidence_score}`;
        } else {
            verdictDisplay.innerHTML = `<span style="color:var(--color-orange); font-weight:700; font-size:14px;">⚠️ UNCERTAIN</span><br>Confidence: ${data.confidence_score}`;
        }

    } catch (error) {
        verdictDisplay.innerHTML = `<span style="color:var(--color-red); font-weight:700;">CONNECTION ERROR</span><br>Check backend server logs.`;
        console.error(error);
    }

    analyzeBtn.disabled = false;
});