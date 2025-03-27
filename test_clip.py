import argparse
import torch
import torch.nn as nn
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
import clip
import torchattacks
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="CIFAR-10", help="Dataset name")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--caption", type=str, default=None, help="Using caption")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load dataset")
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
task_name = args.task_name
# Load dataset
print(f"Loading {task_name} dataset...")

test_dataset, test_loader, classes, NUM_CLASSES = load_test_data(task_name=args.task_name, batch_size=BATCH_SIZE, preprocess=preprocess, data_dir=args.data_dir)
# print(f"Test dataset loaded with {len(test_dataset)} samples.")
print("Processing text prompts...")
text_features = None 

if args.checkpoint:
    print(f"Loading checkpoint: {args.checkpoint}")
    prompter = PromptLearner(clip_model=model, classnames=classes)
    loaded_prompt_data = torch.load(args.checkpoint, map_location=DEVICE)['best_prompt_text']
    # if "token_prefix" in loaded_prompt_data:
    #     del loaded_prompt_data["token_prefix"]

    # if "token_suffix" in loaded_prompt_data:
    #     del loaded_prompt_data["token_suffix"]
    prompter.load_state_dict({'state':loaded_prompt_data}, strict=False)
    text_encoder = TextEncoder(model)
    prompts = prompter()
    text_features = text_encoder(prompts, prompter.tokenized_prompts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"Loaded 'best_prompt_text' as tensor with shape: {loaded_prompt_data.shape}")


elif args.caption:
    print(f"Using single caption '{args.caption}' from command line.")
    text_features = get_text_information(model=model, caption = args.caption, classes=classes, device=DEVICE)

else:
    print("No checkpoint or caption specified.")
    text_features = get_text_information(model=model, classes=classes, device=DEVICE)

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
    print(labels)
    batch_start_time = time.time()
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
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