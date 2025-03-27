import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
from tqdm import tqdm
import torchattacks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = 'ViT-B/32'

BATCH_SIZE = 64 

PGD_EPS = 4/255       
PGD_ALPHA = PGD_EPS/4    
PGD_STEPS = 10        

model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = model.float()
model.cuda()
model.eval() 

print(f"CLIP model '{MODEL_NAME}' loaded.")
print("Input resolution:", model.visual.input_resolution)

test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

cifar100_classes = test_dataset.classes
print(f"CIFAR-100 dataset loaded. Number of classes: {len(cifar100_classes)}")

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
        self.logit_scale = self.clip_model.logit_scale.exp()

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.logit_scale * image_features @ self.text_features.T
        return logits_per_image

clip_wrapper = ClipWrapper(model, text_features).to(DEVICE)

atk = torchattacks.PGD(clip_wrapper, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
# atk.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # <- Không cần thiết vì wrapper đã nhận ảnh chuẩn hóa
print(f"PGD Attack defined: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
print("Note: Attack operates on CLIP's preprocessed (normalized) images.")

clean_correct = 0
robust_correct = 0
total = 0

expected_dtype = model.dtype
print(f"Model dtype: {expected_dtype}")

for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    images = images.to(dtype=expected_dtype)
    # ---------------------------------------------------------

    batch_size_current = images.shape[0]

    with torch.no_grad():
        logits_clean = clip_wrapper(images) 
        predictions_clean = logits_clean.argmax(dim=-1)
        clean_correct += (predictions_clean == labels).sum().item()

    adv_images = atk(images, labels)

    with torch.no_grad():
        logits_adv = clip_wrapper(adv_images)
        predictions_adv = logits_adv.argmax(dim=-1)
        robust_correct += (predictions_adv == labels).sum().item()

    total += batch_size_current

accuracy_clean = 100 * clean_correct / total
accuracy_robust = 100 * robust_correct / total

print("\n--- Evaluation Results ---")
print(f"Total images evaluated: {total}")
print(f"Clean Accuracy (acc): {accuracy_clean:.2f}%")
print(f"Robust Accuracy (rob) against PGD (eps={PGD_EPS}, steps={PGD_STEPS}): {accuracy_robust:.2f}%")