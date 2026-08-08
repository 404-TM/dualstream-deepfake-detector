import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops, ImageEnhance, ImageFile
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm
import warnings

# Suppress PyTorch/PIL warnings during evaluation
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("🔬 Initializing SOTA Empirical Comparison Suite...")

# =============================================================================
# 1. MODEL ARCHITECTURES (Strictly matching the new SOTA Trainers)
# =============================================================================

# --- SINGLE STREAM BASELINE ---
class SingleStreamDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.backbone(x)

# --- SOTA DUAL STREAM PROPOSED ---
class FusionHead(nn.Module):
    def __init__(self, in_features=1792):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )
    def forward(self, rgb_f, noise_f):
        return self.net(torch.cat([rgb_f, noise_f], dim=1))

class DualStreamDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream1 = models.efficientnet_b0(weights=None)
        self.stream1.classifier = nn.Identity() 
        
        self.stream2 = models.mobilenet_v2(weights=None)
        self.stream2.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        ) 
        self.fusion = FusionHead(in_features=1280 + 512)

    def forward(self, rgb, noise):
        return self.fusion(self.stream1(rgb), self.stream2(noise))


# =============================================================================
# 2. FORENSIC & COMPRESSION PIPELINE
# =============================================================================

def apply_clahe(pil_img):
    """Enhances hidden deepfake blending seams before ELA extraction."""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB))

def generate_ela(img, quality=90):
    """Calculates Error Level Analysis dynamically in RAM."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    ela = ImageChops.difference(img, Image.open(buffer))
    ex = ela.getextrema()
    md = max([e[1] for e in ex]) if ex else 1
    scale = min(255.0 / max(md, 1), 50.0)
    return ImageEnhance.Brightness(ela).enhance(scale)

def simulate_compression(pil_img, quality):
    """Simulates WhatsApp/Social Media lossy compression (Proxy for CRF)"""
    if quality == 100: return pil_img
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


# =============================================================================
# 3. EVALUATION ENGINE
# =============================================================================

def load_weights_robustly(model, checkpoint_path, device):
    """Helper to safely load weights regardless of how the dictionary was saved."""
    # Added weights_only=False to bypass PyTorch 2.6+ security restrictions on NumPy metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def evaluate_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Using Device: {device}")

    # Paths to your trained checkpoints (UPDATE THESE IF NEEDED)
    SINGLE_WEIGHTS_PATH = "models/single_stream_baseline.pth"  
    DUAL_WEIGHTS_PATH = "models/sota_best_fusion_model.pth"         

    # 1. Load Models Safely
    single_model = SingleStreamDetector().to(device)
    dual_model = DualStreamDetector().to(device)

    try:
        single_model = load_weights_robustly(single_model, SINGLE_WEIGHTS_PATH, device)
        print("✅ SOTA Single-Stream weights loaded.")
    except Exception as e:
        print(f"❌ Failed to load Single-Stream: {e}")
        return

    try:
        dual_model = load_weights_robustly(dual_model, DUAL_WEIGHTS_PATH, device)
        print("✅ SOTA Dual-Stream weights loaded.")
    except Exception as e:
        print(f"❌ Failed to load Dual-Stream: {e}")
        return

    # Configuration for Compression Trials
    # Quality 100 ~ CRF 0 | Quality 60 ~ CRF 23 | Quality 20 ~ CRF 40
    compression_profiles = {
        "Pristine (CRF 0)": 100,
        "Light (CRF 23)": 60,
        "Severe (CRF 40)": 20
    }
    
    dataset_dir = "./val_data/" 
    if not os.path.exists(dataset_dir):
        print(f"❌ Could not find validation dataset at {dataset_dir}")
        return

    report = {"Single": {}, "Dual": {}}
    roc_data_severe = {}

    print("\n🚀 Commencing Evaluation across Compression Profiles...")

    # Define ImageNet normalization constants
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    for profile_name, quality in compression_profiles.items():
        y_true = []
        y_prob_single = []
        y_prob_dual = []
        
        print(f"\nEvaluating: {profile_name}")
        
        # 1 = Real, 0 = Fake (Matches both SOTA trainers)
        for label_name, label_val in [("real", 1), ("fake", 0)]:
            folder = os.path.join(dataset_dir, label_name)
            if not os.path.exists(folder): continue
            
            filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for filename in tqdm(filenames, desc=f"Processing {label_name.upper()}"):
                filepath = os.path.join(folder, filename)
                try:
                    img = Image.open(filepath).convert('RGB')
                    
                    # 1. Apply Compression Simulation
                    comp_img = simulate_compression(img, quality)
                    
                    # 2. Prepare Single-Stream Input
                    norm_img = TF.normalize(TF.to_tensor(comp_img.resize((224, 224))), mean, std).unsqueeze(0).to(device)
                    
                    # 3. Prepare Dual-Stream Inputs 
                    clahe_img = apply_clahe(comp_img)
                    ela_img = generate_ela(clahe_img)
                    
                    rgb_t = TF.normalize(TF.to_tensor(comp_img.resize((224, 224))), mean, std).unsqueeze(0).to(device)
                    noise_t = TF.normalize(TF.to_tensor(ela_img.resize((224, 224))), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # NOTE: No inversion needed here. SOTA models naturally predict 1=Real, 0=Fake
                        prob_s = torch.sigmoid(single_model(norm_img)).item()
                        prob_d = torch.sigmoid(dual_model(rgb_t, noise_t)).item()
                        
                        y_prob_single.append(prob_s)
                        y_prob_dual.append(prob_d)
                        y_true.append(label_val)
                        
                except Exception as e:
                    pass # Skip corrupted/truncated images safely

        # Calculate Metrics
        y_true = np.array(y_true)
        y_prob_single = np.array(y_prob_single)
        y_prob_dual = np.array(y_prob_dual)
        
        y_pred_single = (y_prob_single >= 0.5).astype(int)
        y_pred_dual = (y_prob_dual >= 0.5).astype(int)

        try:
            auc_s = roc_auc_score(y_true, y_prob_single)
            auc_d = roc_auc_score(y_true, y_prob_dual)
        except ValueError:
            auc_s = auc_d = 0.0
            
        report["Single"][profile_name] = {
            "Acc": accuracy_score(y_true, y_pred_single) * 100,
            "F1": f1_score(y_true, y_pred_single, zero_division=0),
            "AUC": auc_s
        }
        
        report["Dual"][profile_name] = {
            "Acc": accuracy_score(y_true, y_pred_dual) * 100,
            "F1": f1_score(y_true, y_pred_dual, zero_division=0),
            "AUC": auc_d
        }
        
        # Save ROC data for Severe profile to plot later
        if quality == 20:
            fpr_s, tpr_s, _ = roc_curve(y_true, y_prob_single)
            fpr_d, tpr_d, _ = roc_curve(y_true, y_prob_dual)
            roc_data_severe = {"Single": (fpr_s, tpr_s, auc_s), "Dual": (fpr_d, tpr_d, auc_d)}


    # =========================================================================
    # 4. PRINT IEEE-READY TERMINAL REPORT
    # =========================================================================
    print("\n" + "="*85)
    print(" TABLE I. EMPIRICAL PERFORMANCE METRICS ACROSS COMPRESSION PROFILES")
    print("="*85)
    print(f"{'Architecture Profile':<25} | {'Degradation':<18} | {'Binary Acc':<12} | {'ROC AUC':<10} | {'F1-Score'}")
    print("-" * 85)
    for profile in compression_profiles.keys():
        s = report["Single"][profile]
        print(f"{'Single-Stream Baseline':<25} | {profile:<18} | {s['Acc']:>9.2f}%  | {s['AUC']:>9.4f} | {s['F1']:>8.4f}")
    print("-" * 85)
    for profile in compression_profiles.keys():
        d = report["Dual"][profile]
        print(f"{'Proposed Dual-Stream':<25} | {profile:<18} | {d['Acc']:>9.2f}%  | {d['AUC']:>9.4f} | {d['F1']:>8.4f}")
    print("="*85 + "\n")

    # =========================================================================
    # 5. GENERATE FINAL PUBLICATION GRAPHS
    # =========================================================================
    labels = list(compression_profiles.keys())
    x = np.arange(len(labels))
    width = 0.35
    
    # Graph 1: Accuracy Bar Chart
    s_accs = [report["Single"][p]["Acc"] for p in labels]
    d_accs = [report["Dual"][p]["Acc"] for p in labels]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    rects1 = ax.bar(x - width/2, s_accs, width, label='Single-Stream Baseline', color='#dc3545')
    rects2 = ax.bar(x + width/2, d_accs, width, label='Proposed Dual-Stream', color='#007bff')

    ax.set_ylabel('Binary Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Empirical Robustness Under Video Compression', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower left', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
    plt.tight_layout()
    plt.savefig('PUB_Fig1_Accuracy_Comparison.png', bbox_inches='tight')
    print("✅ Saved PUB_Fig1_Accuracy_Comparison.png")
    
    # Graph 2: ROC Curve for Severe Compression
    if roc_data_severe:
        fpr_s, tpr_s, auc_s = roc_data_severe["Single"]
        fpr_d, tpr_d, auc_d = roc_data_severe["Dual"]
        
        plt.figure(figsize=(7, 6), dpi=300)
        plt.plot(fpr_d, tpr_d, color='#007bff', lw=2.5, label=f'Dual-Stream (AUC = {auc_d:.3f})')
        plt.plot(fpr_s, tpr_s, color='#dc3545', lw=2.5, linestyle='--', label=f'Single-Stream (AUC = {auc_s:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve at Severe Compression (CRF-40)', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('PUB_Fig2_Severe_ROC.png', bbox_inches='tight')
        print("✅ Saved PUB_Fig2_Severe_ROC.png")

    print("🎉 SOTA Evaluation Complete! Your true publication metrics are ready.")

if __name__ == "__main__":
    evaluate_models()