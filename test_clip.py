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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--prompt", type=str, default=None, help="Custom prompt (WARNING: Likely incompatible with PGD evaluation if used)")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = 'ViT-B/32'
BATCH_SIZE = 64
PGD_EPS = 4/255
PGD_ALPHA = PGD_EPS/4
PGD_STEPS = 10

print("Loading CLIP model...")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model = model.float() 
model.eval()

print(f"CLIP model '{MODEL_NAME}' loaded .")
print("Input resolution:", model.visual.input_resolution)

# Load dataset
print("Loading CIFAR-10 dataset...")
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

cifar10_classes = test_dataset.classes
NUM_CLASSES = len(cifar10_classes)
print(f"CIFAR-10 dataset loaded. Number of classes: {NUM_CLASSES}")

print("Processing text prompts...")
text_features = None 

def get_default_text_features(clip_model, class_names, device):
    print("Generating text features using default class name prompts...")
    text_descriptions = [f"{class_name}" for class_name in class_names]
    text_tokens = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_tokens)
        features /= features.norm(dim=-1, keepdim=True)
    print(f"Generated default text features with shape: {features.shape}")
    return features.float()

if args.checkpoint:
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

    if 'best_prompt_text' in checkpoint:
        loaded_prompt_data = checkpoint['best_prompt_text']

        if isinstance(loaded_prompt_data, torch.Tensor):
            print(f"Loaded 'best_prompt_text' as tensor with shape: {loaded_prompt_data.shape}")
            # KIỂM TRA SHAPE!
            if loaded_prompt_data.shape[0] == NUM_CLASSES:
                print("Shape matches number of classes. Using loaded text features.")
                text_features = loaded_prompt_data.to(device=DEVICE).float() # Đảm bảo đúng device và dtype
            else:
                print(f"WARNING: Loaded tensor shape {loaded_prompt_data.shape} does not match number of classes ({NUM_CLASSES}).")
                print("Falling back to default class name prompts for evaluation.")
                text_features = get_default_text_features(model, cifar10_classes, DEVICE)
        elif isinstance(loaded_prompt_data, str):
            print(f"WARNING: Loaded 'best_prompt_text' is a string: '{loaded_prompt_data}'.")
            print("Single string prompt is incompatible with multi-class PGD evaluation.")
            print("Falling back to default class name prompts.")
            text_features = get_default_text_features(model, cifar10_classes, DEVICE)
        else:
            print("WARNING: 'best_prompt_text' in checkpoint has unexpected format.")
            print("Falling back to default class name prompts.")
            text_features = get_default_text_features(model, cifar10_classes, DEVICE)
    else:
        print("Warning: 'best_prompt_text' not found in checkpoint.")
        print("Falling back to default class name prompts.")
        text_features = get_default_text_features(model, cifar10_classes, DEVICE)

elif args.prompt:
    print(f"WARNING: Using single custom prompt '{args.prompt}' from command line.")
    print("Single custom prompt is incompatible with multi-class PGD evaluation.")
    print("Falling back to default class name prompts.")
    text_features = get_default_text_features(model, cifar10_classes, DEVICE)

else:
    print("No checkpoint or custom prompt specified.")
    text_features = get_default_text_features(model, cifar10_classes, DEVICE)

if text_features is None or text_features.shape[0] != NUM_CLASSES:
     raise ValueError(f"Failed to obtain valid text features for {NUM_CLASSES} classes. Final shape: {text_features.shape if text_features is not None else 'None'}")

print(f"Using final text features with shape: {text_features.shape}")

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
    try:
        with torch.no_grad():
            logits_clean = clip_custom(images)
            predictions_clean = logits_clean.argmax(dim=-1)
            batch_clean_correct = (predictions_clean == labels).sum().item()
            clean_correct_total += batch_clean_correct
    except Exception as e:
        print(f"\nError during clean evaluation on batch {i}: {e}")
        print(f"Image batch dtype: {images.dtype}, shape: {images.shape}")
        print(f"Text features dtype: {clip_custom.text_features.dtype}, shape: {clip_custom.text_features.shape}")
        raise

    try:
        adv_images = atk(images, labels)
    except Exception as e:
        print(f"\nError during PGD attack generation on batch {i}: {e}")
        print(f"Image batch dtype: {images.dtype}, shape: {images.shape}")
        print(f"Label batch dtype: {labels.dtype}, shape: {labels.shape}")
        print(f"Text features dtype: {clip_custom.text_features.dtype}, shape: {clip_custom.text_features.shape}")
        # Thêm gợi ý debug CUDA
        print("\nPotential CUDA Error. Consider running with CUDA_LAUNCH_BLOCKING=1 python your_script.py ... for more specific error location.")
        raise

    try:
        with torch.no_grad():
            logits_adv = clip_custom(adv_images)
            predictions_adv = logits_adv.argmax(dim=-1)
            batch_robust_correct = (predictions_adv == labels).sum().item()
            robust_correct_total += batch_robust_correct
    except Exception as e:
        print(f"\nError during robust evaluation on batch {i}: {e}")
        print(f"Adv Image batch dtype: {adv_images.dtype}, shape: {adv_images.shape}")
        print(f"Text features dtype: {clip_custom.text_features.dtype}, shape: {clip_custom.text_features.shape}")
        raise

    total_images += batch_size_current

    # --- Calculate and Print Batch Accuracy ---
    batch_acc = 100 * batch_clean_correct / batch_size_current
    batch_rob = 100 * batch_robust_correct / batch_size_current
    batch_end_time = time.time()
    if i % 10 == 0 or i == len(test_loader) - 1: # In batch đầu, cuối và mỗi 10 batch
        print(f"Batch {i+1}/{len(test_loader)} | Size: {batch_size_current} | Time: {batch_end_time - batch_start_time:.2f}s | Clean Acc: {batch_acc:.2f}% | Robust Acc: {batch_rob:.2f}%")

# --- Final Results ---
if total_images > 0:
    accuracy_clean = 100 * clean_correct_total / total_images
    accuracy_robust = 100 * robust_correct_total / total_images

    print("\n--- Final Evaluation Results ---")
    print(f"Total images evaluated: {total_images}")
    print(f"Overall Clean Accuracy: {accuracy_clean:.2f}%")
    print(f"Overall Robust Accuracy against PGD (eps={PGD_EPS}, steps={PGD_STEPS}): {accuracy_robust:.2f}%")
else:
      print("\nNo images were evaluated.")