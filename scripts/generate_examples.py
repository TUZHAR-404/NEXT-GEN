from pathlib import Path
import sys
import torch
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.resnet_model import create_resnet_model

ROOT = Path('D:/AIMS-DTU/COVID_19_dataset')
TEST = ROOT / 'test'
ckpt = Path('checkpoints/resnet_best.pt')
out_dir = Path('results/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = create_resnet_model(num_classes=3, pretrained=False)
if ckpt.exists():
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(sd['model_state_dict'])
model.eval()

# Transform for model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load ImageFolder
dataset = datasets.ImageFolder(TEST, transform=None)
class_names = dataset.classes
print('Classes:', class_names)
# Build indices
from torch.utils.data import DataLoader

normal_idx = class_names.index('Normal') if 'Normal' in class_names else None

found_normal = None
found_disease = None

for path, target in dataset.samples:
    img = Image.open(path).convert('RGB')
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    # Choose normal where predicted == Normal
    if found_normal is None and normal_idx is not None and pred == normal_idx:
        found_normal = (path, class_names[pred], conf)
    # Choose disease where predicted != Normal
    if found_disease is None and (normal_idx is None or pred != normal_idx):
        found_disease = (path, class_names[pred], conf)
    if found_normal and found_disease:
        break

try:
    font = ImageFont.truetype('arial.ttf', 24)
except:
    font = ImageFont.load_default()

for item, name in [(found_normal, 'normal_detected.png'), (found_disease, 'disease_detected.png')]:
    if item is None:
        print('Not found', name)
        continue
    path, pred_label, conf = item
    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img)
    text = f"Pred: {pred_label} ({conf*100:.1f}%)"
    # rectangle behind text
    try:
        w, h = draw.textsize(text, font=font)
    except Exception:
        try:
            w, h = font.getsize(text)
        except Exception:
            bbox = draw.textbbox((0,0), text, font=font)
            w = bbox[2]-bbox[0]
            h = bbox[3]-bbox[1]
    draw.rectangle([(5,5),(5+w+8,5+h+8)], fill=(0,0,0,160))
    draw.text((9,8), text, fill=(255,255,255), font=font)
    out_path = out_dir / name
    img.save(out_path)
    print('Saved', out_path)
