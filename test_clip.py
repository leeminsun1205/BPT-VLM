import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
from tqdm import tqdm
import torchattacks

# --- 1. Cài đặt và Tham số ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Chọn mô hình CLIP (các lựa chọn khác: 'RN50', 'RN101', 'ViT-L/14', ...)
MODEL_NAME = 'ViT-B/32'

BATCH_SIZE = 64 # Giảm nếu gặp lỗi hết bộ nhớ (Out of Memory - OOM)

# Tham số tấn công PGD
# Lưu ý: Ảnh đầu vào của CLIP đã được chuẩn hóa (normalize).
# Các giá trị epsilon và alpha này hoạt động trên không gian ảnh đã chuẩn hóa.
# Các giá trị như 8/255 thường dành cho ảnh trong khoảng [0, 1].
# Chúng ta sẽ dùng giá trị nhỏ hơn cho không gian đã chuẩn hóa.
PGD_EPS = 4/255        # Cường độ tấn công tối đa (epsilon)
PGD_ALPHA = PGD_EPS/4    # Kích thước bước tấn công (alpha)
PGD_STEPS = 10        # Số bước tấn công

# --- 2. Tải mô hình CLIP và Bộ tiền xử lý ---
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.cuda()
model.eval() # Chuyển sang chế độ đánh giá

print(f"CLIP model '{MODEL_NAME}' loaded.")
print("Input resolution:", model.visual.input_resolution)
# print("Preprocessing transform:", preprocess) # In ra để xem các bước chuẩn hóa

# --- 3. Tải và Chuẩn bị Dataset CIFAR-100 ---
# Sử dụng bộ tiền xử lý của CLIP cho dữ liệu test
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

cifar100_classes = test_dataset.classes
print(f"CIFAR-100 dataset loaded. Number of classes: {len(cifar100_classes)}")

# --- 4. Chuẩn bị Text Prompts cho CLIP ---
# Tạo các câu mô tả (prompts) cho mỗi lớp
text_descriptions = [f"{class_name}" for class_name in cifar100_classes]
text_tokens = clip.tokenize(text_descriptions).to(DEVICE)

# Mã hóa các câu mô tả thành vector đặc trưng (text features)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    # Chuẩn hóa text features
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Text features encoded and normalized.")

# --- 5. Định nghĩa Wrapper Model cho PGD ---
# PGD trong torchattacks thường tối ưu hóa CrossEntropyLoss trên logits.
# CLIP dùng cosine similarity. Wrapper này tính similarity và coi đó là 'logits'.
class ClipWrapper(nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = text_features
        # Sử dụng logit_scale nếu có (ảnh hưởng đến độ lớn của 'logits')
        try:
            self.logit_scale = self.clip_model.logit_scale.exp()
        except AttributeError:
            print("Warning: logit_scale not found in model. Using default scale.")
            # Nếu không có logit_scale, có thể dùng giá trị mặc định hoặc bỏ qua
            # self.logit_scale = torch.tensor(100.0).to(DEVICE) # Ví dụ giá trị mặc định
            self.logit_scale = 1.0 # Hoặc đơn giản là 1.0 nếu không muốn scale

    def forward(self, images):
        # Mã hóa ảnh
        image_features = self.clip_model.encode_image(images)
        # Chuẩn hóa image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Tính cosine similarity (đây sẽ là 'logits' cho PGD)
        # Áp dụng logit_scale để giống với cách tính của CLIP
        logits_per_image = self.logit_scale * image_features @ self.text_features.T
        return logits_per_image

# Khởi tạo wrapper
clip_wrapper = ClipWrapper(model, text_features).to(DEVICE)

# --- 6. Định nghĩa Tấn công PGD ---
# Sử dụng wrapper model với PGD
# torchattacks giả định đầu vào nằm trong khoảng [0, 1] theo mặc định,
# nhưng ở đây đầu vào là ảnh đã qua preprocess của CLIP (đã chuẩn hóa).
# PGD sẽ hoạt động trên không gian ảnh đã chuẩn hóa này.
atk = torchattacks.PGD(clip_wrapper, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
# atk.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # <- Không cần thiết vì wrapper đã nhận ảnh chuẩn hóa
print(f"PGD Attack defined: eps={PGD_EPS}, alpha={PGD_ALPHA}, steps={PGD_STEPS}")
print("Note: Attack operates on CLIP's preprocessed (normalized) images.")

# --- 7. Vòng lặp Đánh giá ---
clean_correct = 0
robust_correct = 0
total = 0

for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    batch_size_current = images.shape[0]

    # --- 7a. Đánh giá trên ảnh sạch (Clean Accuracy - acc) ---
    with torch.no_grad():
        # Sử dụng wrapper để lấy logits (similarities)
        logits_clean = clip_wrapper(images)
        predictions_clean = logits_clean.argmax(dim=-1)
        clean_correct += (predictions_clean == labels).sum().item()

    # --- 7b. Tạo ảnh bị tấn công (Adversarial Images) ---
    # PGD cần gradient, nên không dùng torch.no_grad() ở đây
    # atk nhận ảnh đã chuẩn hóa và trả về ảnh tấn công cũng đã chuẩn hóa
    adv_images = atk(images, labels)

    # --- 7c. Đánh giá trên ảnh bị tấn công (Robust Accuracy - rob) ---
    with torch.no_grad():
        # Sử dụng wrapper để lấy logits (similarities) cho ảnh tấn công
        logits_adv = clip_wrapper(adv_images)
        predictions_adv = logits_adv.argmax(dim=-1)
        robust_correct += (predictions_adv == labels).sum().item()

    total += batch_size_current

# --- 8. Tính toán và In kết quả ---
accuracy_clean = 100 * clean_correct / total
accuracy_robust = 100 * robust_correct / total

print("\n--- Evaluation Results ---")
print(f"Total images evaluated: {total}")
print(f"Clean Accuracy (acc): {accuracy_clean:.2f}%")
print(f"Robust Accuracy (rob) against PGD (eps={PGD_EPS}, steps={PGD_STEPS}): {accuracy_robust:.2f}%")