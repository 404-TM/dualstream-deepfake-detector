import os
import io
import cv2
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops, ImageEnhance
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

print("⚙️ Initializing Backend v13.0 (Multi-Band CLAHE Edition)...")

app = FastAPI(title="ASEP Deepfake API - Multi-Band Forensics")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend"), name="frontend_static")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Compatibility endpoint used by the frontend single-frame extractor.
    Returns normalized scores (0.0-1.0) for the frontend gauges plus decision info.
    """
    try:
        pil_img = Image.open(io.BytesIO(await file.read())).convert('RGB')
        cropped_face, face_box = detect_face_and_crop(pil_img)

        ai_score = predict_multi_band_ai(cropped_face)
        anomaly_ratio = get_structural_anomaly_ratio(pil_img, face_box)

        final_risk = ai_score
        if ai_score > 40.0 and anomaly_ratio > 1.8:
            final_risk += 20.0
        final_risk = np.clip(final_risk, 0.0, 100.0)

        if final_risk > 55.0:
            final_decision = "FAKE"
            display_score = final_risk
        elif final_risk < 40.0:
            final_decision = "REAL"
            display_score = 100.0 - final_risk
        else:
            final_decision = "UNCERTAIN (Suspicious Web Compression)"
            display_score = final_risk

        # Return normalized floats for frontend gauges
        normalized = float(final_risk) / 100.0
        return {
            "rgb_score": normalized,
            "noise_score": normalized,
            "decision": final_decision,
            "confidence_score": f"{display_score:.2f}%"
        }
    except Exception as e:
        return {"error": f"Predict failed: {str(e)}"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. THE ARCHITECTURE (92.40% Dual-Stream Brain)
class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1792, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(p=0.60), nn.Linear(512, 1))
    def forward(self, f1, f2):
        return self.net(torch.cat([f1, f2], dim=1))

class DualStreamDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream1 = models.efficientnet_b0(weights=None)
        self.stream1.classifier = nn.Identity()
        self.stream2 = models.mobilenet_v2(weights=None)
        self.stream2.classifier = nn.Sequential(nn.Flatten(), nn.Linear(1280, 512), nn.ReLU(inplace=True))
        self.fusion  = FusionHead()
    def forward(self, rgb, noise):
        return self.fusion(self.stream1(rgb), self.stream2(noise))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualStreamDetector()
brain_path = "models/clean_master_fusion_model.pth" 

try:
    model.load_state_dict(torch.load(brain_path, map_location=device), strict=True)
    model.to(device)
    model.eval()
    print("✅ Brain Loaded Successfully!")
except Exception as e:
    print(f"🚨 ERROR: Could not load {brain_path}. {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. THE 1,000,000% STABLE FORENSIC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def detect_face_and_crop(pil_img):
    open_cv_image = np.array(pil_img) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    h_img, w_img = open_cv_image.shape[:2]
    if len(faces) == 0: 
        x, y, w, h = int(w_img*0.2), int(h_img*0.2), int(w_img*0.6), int(h_img*0.6)
    else:
        x, y, w, h = faces[0]
        
    pad_w, pad_h = int(w*0.2), int(h*0.2)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)
    
    return pil_img.crop((x1, y1, x2, y2)), (x, y, w, h)

def apply_clahe_enhancement(pil_img):
    """🛠️ HACK 13: CLAHE. Rips through web compression to expose hidden deepfake stitching."""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)

def generate_multi_band_ela(enhanced_img, quality):
    """Generates ELA at specific frequencies."""
    buffer = io.BytesIO()
    enhanced_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    ela_image = ImageChops.difference(enhanced_img, compressed)
    ex = ela_image.getextrema()
    md = max([e[1] for e in ex]) if ex else 1
    
    scale = min(255.0 / max(md, 1), 50.0) 
    ela_boosted = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_boosted

def get_structural_anomaly_ratio(full_pil_img, face_box):
    """Calculates if the face degrades differently than the background."""
    x, y, w, h = face_box
    buffer = io.BytesIO()
    full_pil_img.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)
    
    ela_img = ImageChops.difference(full_pil_img, Image.open(buffer))
    ela_array = np.array(ela_img.convert('L'))

    face_ela = ela_array[y:y+h, x:x+w]
    bg_mask = np.ones_like(ela_array, dtype=bool)
    bg_mask[y:y+h, x:x+w] = False
    bg_ela = ela_array[bg_mask]

    face_mean = np.mean(face_ela) + 0.1
    bg_mean = np.mean(bg_ela) + 0.1
    
    ratio = max(face_mean / bg_mean, bg_mean / face_mean)
    return ratio

def predict_multi_band_ai(base_img):
    """Feeds the AI model 3 different compression bands for maximum stability."""
    clahe_img = apply_clahe_enhancement(base_img)
    qualities = [90, 80, 70] # Multi-Band Resonance
    probabilities = []
    
    with torch.no_grad():
        for q in qualities:
            noise_img = generate_multi_band_ela(clahe_img, q)
            
            rgb_t = TF.normalize(TF.to_tensor(clahe_img.resize((224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            noise_t = TF.normalize(TF.to_tensor(noise_img.resize((224, 224))), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
            logits = model(rgb_t.unsqueeze(0).to(device), noise_t.unsqueeze(0).to(device))
            probabilities.append(torch.sigmoid(logits).item() * 100)
            
    # Return the highest suspicion across the 3 bands
    return max(probabilities)

# ─────────────────────────────────────────────────────────────────────────────
# 3. THE API ENDPOINTS (Cleaned & Stabilized)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        pil_img = Image.open(io.BytesIO(await file.read())).convert('RGB')
        cropped_face, face_box = detect_face_and_crop(pil_img)
        
        # 1. Multi-Band AI Prediction
        ai_score = predict_multi_band_ai(cropped_face)
        
        # 2. Structural Anomaly Check (Only applies if the AI is suspicious)
        anomaly_ratio = get_structural_anomaly_ratio(pil_img, face_box)
        final_risk = ai_score
        
        # If the AI sees a deepfake AND the background/face mismatch is high, boost confidence
        if ai_score > 40.0 and anomaly_ratio > 1.8:
            final_risk += 20.0
            
        final_risk = np.clip(final_risk, 0.0, 100.0)

        # 3. Stable Decision Logic
        if final_risk > 55.0:          
            final_decision = "FAKE"
            display_score = final_risk
        elif final_risk < 40.0:        
            final_decision = "REAL"
            display_score = 100.0 - final_risk 
        else:
            final_decision = "UNCERTAIN (Suspicious Web Compression)"
            display_score = final_risk
        
        print(f"🛡️ V13 DIAGNOSTIC -> Raw AI: {ai_score:.1f}% | Anomaly Ratio: {anomaly_ratio:.2f} | Final: {final_risk:.1f}%")

        return {
            "status": "success",
            "decision": final_decision,
            "confidence_score": f"{display_score:.2f}%",
            "analyzed_type": "Static Image"
        }
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release(); os.remove(temp_video_path)
        return {"error": "Could not read video file."}

    target_frames = set([int(total_frames * i) for i in [0.1, 0.3, 0.5, 0.7, 0.9]])
    frame_scores = []
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 
        if current_frame in target_frames:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped_face, face_box = detect_face_and_crop(pil_frame)
            
            ai_score = predict_multi_band_ai(cropped_face)
            anomaly_ratio = get_structural_anomaly_ratio(pil_frame, face_box)
            
            risk = ai_score
            if ai_score > 40.0 and anomaly_ratio > 1.8:
                risk += 20.0
                
            frame_scores.append(np.clip(risk, 0.0, 100.0))
            
        current_frame += 1
        if current_frame > max(target_frames): break

    cap.release()
    os.remove(temp_video_path) 
    if not frame_scores: return {"error": "Failed to extract frames."}

    avg_risk = sum(frame_scores) / len(frame_scores)
    
    if avg_risk > 50.0:          
        final_decision = "FAKE"
        display_score = avg_risk
    elif avg_risk < 45.0:        
        final_decision = "REAL"
        display_score = 100.0 - avg_risk 
    else:
        final_decision = "UNCERTAIN (Suspicious Web Compression)"
        display_score = avg_risk
    
   # CHANGE THIS:
    # return {"status": "success", "decision": final_decision, "confidence_score": f"{display_score:.2f}%"}
    
    # TO THIS:
    return {
        "status": "success", 
        "decision": final_decision, 
        "confidence_score": f"{display_score:.2f}%",
        "frame_scores": frame_scores  # <--- Now the frontend can see the frame data!
    }

@app.get("/")
def home(): return {"message": "ASEP Deepfake API - Multi-Band CLAHE Edition is Online."}