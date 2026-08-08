console.log("Dual-Stream Content Script Loaded.");

// This function does the heavy lifting of capturing a frame from a video element
function captureVideoFrame() {
    // 1. Find the first <video> element on the current webpage
    const video = document.querySelector('video');
    
    if (!video) {
        console.log("No video element found on this page.");
        return null;
    }

    console.log(`Video found! Size: ${video.videoWidth}x${video.videoHeight}`);

    // 2. Create a hidden, in-memory HTML5 Canvas element
    const canvas = document.createElement('canvas');
    
    // 3. Match the canvas dimensions exactly to the raw video resolution
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // 4. Get the 2D rendering context to draw on the canvas
    const ctx = canvas.getContext('2d');
    
    // 5. Draw the CURRENT frame of the video onto our hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // 6. Convert the canvas drawing into a JPEG Base64 Data URL string
    // We use 'image/jpeg' with 0.9 quality because JPEG compression artifacts 
    // mimic the digital noise your ML team is trying to analyze!
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
    
    console.log("Frame successfully captured as JPEG Data URL!");
    return imageDataUrl;
}

// 7. Listen for instructions from the popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "CAPTURE_FRAME") {
        const frameData = captureVideoFrame();
        
        if (frameData) {
            // Send the image data back to whoever requested it (the popup)
            sendResponse({ status: "success", data: frameData });
        } else {
            sendResponse({ status: "error", message: "No video found on page." });
        }
    }
    return true; // Keeps the communication line open for asynchronous response
});