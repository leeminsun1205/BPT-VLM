import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
# from tqdm import tqdm # Removed tqdm to allow clear batch printing
import torchattacks
import time # Added for batch timing (optional)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = 'ViT-B/32'

BATCH_SIZE = 64

PGD_EPS = 4/255
PGD_ALPHA = PGD_EPS/4
PGD_STEPS = 10

model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = model.float()
# model.cuda() # Not strictly necessary if device=DEVICE is cuda and model loaded there or moved by .float()
model.eval()

print(f"CLIP model '{MODEL_NAME}' loaded.")
print("Input resolution:", model.visual.input_resolution)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
# Reduce num_workers if encountering issues
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

cifar100_classes = test_dataset.classes
print(f"CIFAR-100 dataset loaded. Number of classes: {len(cifar100_classes)}")

# Using class names directly as prompts
text_descriptions = [f"{class_name}" for class_name in cifar100_classes]
text_tokens = clip.tokenize(text_descriptions).to(DEVICE)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Text features encoded and normalized.")

class ClipWrapper(nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features
        # Ensure logit_scale is handled correctly after .float()
        self.logit_scale = self.clip_model.logit_scale.exp()

    def forward(self, images):
        # Model and images should both be float32 now
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale * image_features @ self.text_features.T
        return logits_per_image

clip_wrapper = ClipWrapper(model, text_features).to(DEVICE)

atk = torchattacks.PGD(clip_wrapper, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
print(f"PGD Attack defined: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
print("Note: Attack operates on CLIP's preprocessed (normalized) images.")

clean_correct_total = 0 # Renamed for clarity
robust_correct_total = 0 # Renamed for clarity
total_images = 0 # Renamed for clarity

# expected_dtype = model.dtype # Not needed if model is float32 and data is float32
# print(f"Model dtype: {expected_dtype}") # Not needed if model is float32

print("\nStarting evaluation loop...")
# Removed tqdm wrapper: for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
for i, (images, labels) in enumerate(test_loader):
    batch_start_time = time.time()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    # No need to manually convert images.to(dtype) if model and dataloader use float32
    # images = images.to(dtype=expected_dtype)

    batch_size_current = images.shape[0]
    batch_clean_correct = 0
    batch_robust_correct = 0

    # --- Clean Evaluation ---
    with torch.no_grad():
        logits_clean = clip_wrapper(images)
        predictions_clean = logits_clean.argmax(dim=-1)
        batch_clean_correct = (predictions_clean == labels).sum().item()
        clean_correct_total += batch_clean_correct

    # --- Adversarial Attack Generation ---
    adv_images = atk(images, labels)

    # --- Robust Evaluation ---
    with torch.no_grad():
        logits_adv = clip_wrapper(adv_images)
        predictions_adv = logits_adv.argmax(dim=-1)
        batch_robust_correct = (predictions_adv == labels).sum().item()
        robust_correct_total += batch_robust_correct

    total_images += batch_size_current

    # --- Calculate and Print Batch Accuracy ---
    batch_acc = 100 * batch_clean_correct / batch_size_current
    batch_rob = 100 * batch_robust_correct / batch_size_current
    batch_end_time = time.time()
    print(f"Batch {i+1}/{len(test_loader)} | Size: {batch_size_current} | Time: {batch_end_time - batch_start_time:.2f}s | Clean Acc: {batch_acc:.2f}% | Robust Acc: {batch_rob:.2f}%")


# --- Final Results ---
if total_images > 0:
    accuracy_clean = 100 * clean_correct_total / total_images
    accuracy_robust = 100 * robust_correct_total / total_images

    print("\n--- Final Evaluation Results ---")
    print(f"Total images evaluated: {total_images}")
    print(f"Overall Clean Accuracy (acc): {accuracy_clean:.2f}%")
    print(f"Overall Robust Accuracy (rob) against PGD (eps={PGD_EPS}, steps={PGD_STEPS}): {accuracy_robust:.2f}%")
else:
    print("\nNo images were evaluated.")