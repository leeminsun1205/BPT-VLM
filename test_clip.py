import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
import torchattacks
import time
from utils import ClipCustom

# Xử lý đối số đầu vào
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = 'ViT-B/32'
BATCH_SIZE = 64
PGD_EPS = 4/255
PGD_ALPHA = PGD_EPS/4
PGD_STEPS = 10

# Load mô hình CLIP
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = model.float()
model.eval()

print(f"CLIP model '{MODEL_NAME}' loaded.")
print("Input resolution:", model.visual.input_resolution)

# Load dataset
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

cifar10_classes = test_dataset.classes
print(f"CIFAR-10 dataset loaded. Number of classes: {len(cifar10_classes)}")

# Xử lý text prompt
if args.checkpoint:
    # Load checkpoint từ file
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    
    print("Checkpoint keys:", checkpoint.keys())  # Kiểm tra các key có trong checkpoint

    prompt_clip = ClipCustom(model, None).to(DEVICE)

    # Kiểm tra xem có best_prompt_text không
    if 'best_prompt_text' in checkpoint:
        best_prompt = checkpoint['best_prompt_text']
        if isinstance(best_prompt, torch.Tensor):
            print("Loaded best prompt as tensor, skipping tokenization.")
            text_features = best_prompt.to(DEVICE)
        else:
            print(f"Loaded best prompt as text: {best_prompt}")
            text_tokens = clip.tokenize([best_prompt]).to(DEVICE)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                

elif args.prompt:
    # Sử dụng prompt tùy chỉnh do người dùng nhập vào
    print(f"Using custom prompt: {args.prompt}")
    text_tokens = clip.tokenize([args.prompt]).to(DEVICE)

else:
    # Dùng tên lớp mặc định làm prompt
    text_descriptions = [f"{class_name}" for class_name in cifar10_classes]
    text_tokens = clip.tokenize(text_descriptions).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Text features encoded and normalized.")

clip_custom = ClipCustom(model, text_features).to(DEVICE)

atk = torchattacks.PGD(clip_custom, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
print(f"PGD Attack defined: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
print("Note: Attack operates on CLIP's preprocessed (normalized) images.")

clean_correct_total = 0
robust_correct_total = 0
total_images = 0

print("\nStarting evaluation loop...")
for i, (images, labels) in enumerate(test_loader):
    batch_start_time = time.time()
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    batch_size_current = images.shape[0]
    batch_clean_correct = 0
    batch_robust_correct = 0

    # --- Clean Evaluation ---
    with torch.no_grad():
        logits_clean = clip_custom(images)
        predictions_clean = logits_clean.argmax(dim=-1)
        batch_clean_correct = (predictions_clean == labels).sum().item()
        clean_correct_total += batch_clean_correct

    # --- Adversarial Attack Generation ---
    adv_images = atk(images, labels)

    # --- Robust Evaluation ---
    with torch.no_grad():
        logits_adv = clip_custom(adv_images)
        predictions_adv = logits_adv.argmax(dim=-1)
        batch_robust_correct = (predictions_adv == labels).sum().item()
        robust_correct_total += batch_robust_correct

    total_images += batch_size_current

    # --- Calculate and Print Batch Accuracy ---
    batch_acc = 100 * batch_clean_correct / batch_size_current
    batch_rob = 100 * batch_robust_correct / batch_size_current
    batch_end_time = time.time()
    if i % 10 == 0:
        print(f"Batch {i+1}/{len(test_loader)} | Size: {batch_size_current} | Time: {batch_end_time - batch_start_time:.2f}s | Clean Acc: {batch_acc:.2f}% | Robust Acc: {batch_rob:.2f}%")

# --- Final Results ---
accuracy_clean = 100 * clean_correct_total / total_images
accuracy_robust = 100 * robust_correct_total / total_images

print("\n--- Final Evaluation Results ---")
print(f"Total images evaluated: {total_images}")
print(f"Overall Clean Accuracy: {accuracy_clean:.2f}%")
print(f"Overall Robust Accuracy against PGD (eps={PGD_EPS}, steps={PGD_STEPS}): {accuracy_robust:.2f}%")
