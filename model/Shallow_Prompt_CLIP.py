import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchvision.datasets import CIFAR100
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from model.shallow_encoder import TextEncoder,VisionEncoder
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
import copy # Thêm import copy

class PromptCLIP_Shallow:
    def __init__(self,task_name,cfg):
        self.task_name = task_name
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.batch_size = cfg["batch_size"]
        self.k_shot = cfg["k_shot"]
        self.seed = cfg["seed"]
        self.num_call = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.backbone,device=self.device)
        self.load_dataset()
        self.loss = []
        self.acc = []
        # Text Encoder
        self.n_prompt_tokens_L = cfg["n_prompt_tokens_L"]
        self.intrinsic_dim_L = cfg["intrinsic_dim_L"]
        self.ctx_dim_L = self.model.ln_final.weight.shape[0]
        self.text_encoder = TextEncoder(self.model)

        # Image Encoder
        self.n_prompt_tokens_V = cfg["n_prompt_tokens_V"]
        self.ctx_dim_V = self.model.visual.width
        self.intrinsic_dim_V = cfg["intrinsic_dim_V"]
        self.image_encoder = VisionEncoder(self.model)
        self.image_encoder.n_prompt_tokens_V = self.n_prompt_tokens_V

        self.loss_type = cfg["loss_type"]
        self.init_prompt = None
        self.imsize = self.image_encoder.input_resolution
        self.logit_scale = self.model.logit_scale
        self.dtype = self.model.dtype
        self.best_prompt_text = None
        self.best_prompt_image = None
        self.best_accuracy = 0
        self.min_loss = None
        self.loss = []
        self.test_every = cfg["test_every"] if self.parallel else cfg["test_every"]*self.popsize
        self.sigma = cfg["sigma"]

        # --- Thêm tham số PGD ---
        self.pgd_eps = cfg.get("pgd_eps", 8/255) # Ngưỡng nhiễu (epsilon)
        self.pgd_alpha = cfg.get("pgd_alpha", 2/255) # Kích thước bước (alpha)
        self.pgd_steps = cfg.get("pgd_steps", 10) # Số bước PGD
        # Xác định min/max cho clip dựa trên normalization của CLIP
        # Giá trị này là xấp xỉ, bạn có thể tính chính xác hơn nếu cần
        self.clip_min = (0.0 - np.mean([0.48145466, 0.4578275, 0.40821073])) / np.mean([0.26862954, 0.26130258, 0.27577711])
        self.clip_max = (1.0 - np.mean([0.48145466, 0.4578275, 0.40821073])) / np.mean([0.26862954, 0.26130258, 0.27577711])
        # ------------------------

        # Lauguage Linear Layer
        self.linear_L = torch.nn.Linear(self.intrinsic_dim_L, self.n_prompt_tokens_L * self.ctx_dim_L,
                                      bias=False,device=self.device,dtype=self.dtype)
        embedding = self.model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_L.parameters():
            torch.nn.init.normal_(p, mu, std)
        # Vision Linear Layer
        self.linear_V = torch.nn.Linear(self.intrinsic_dim_V, self.n_prompt_tokens_V * self.ctx_dim_V,
                                        bias=False, device=self.device, dtype=self.dtype)
        conv = self.model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
        #mu = 0.0
        mu = mu_hat*3072/self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072/self.intrinsic_dim_V) * self.sigma
        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)

    # ... (các phương thức khác giữ nguyên: get_text_information, get_image_information, generate_text_prompts, generate_visual_prompts, metric) ...

    def get_text_information(self,caption=None):
        # classification task - caption - None
        # refcoco ask - caption - str
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
        if caption is None:
            classnames = [name.replace("_", " ").replace("-"," ") for name in self.classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts= torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":self.n_cls, "n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,
                       "batch_size":self.batch_size,"pop_size":self.popsize,"parallel":self.parallel}
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(self.device)
            with torch.no_grad():
                init_pattern_embedding = self.model.token_embedding(tokenized_pattern_prompts).type(self.dtype)
            context = {"n_cls":1,"n_prompt_tokens_L":self.n_prompt_tokens_L,
                       "init_pattern_embedding":init_pattern_embedding, "tokenized_pattern_prompts":tokenized_pattern_prompts,"batch_size":self.batch_size,
                       "pop_size":self.popsize,"parallel":self.parallel}
        return context

    def get_image_information(self):
        context = {"n_prompt_tokens_V": self.n_prompt_tokens_V,
                   "batch_size": self.batch_size, "pop_size": self.popsize, "parallel": self.parallel}
        return context

    def generate_text_prompts(self,intrinsic_vectors):
        prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector, device=self.device, dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_L(z).reshape(self.n_prompt_tokens_L, -1)
            if self.init_prompt is not None:
                z = z + self.init_prompt  # Az + p_0

            prompt_list.append(z)
        return prompt_list

    def generate_visual_prompts(self,intrinsic_vectors):
        visual_prompt_list = []
        for vector in intrinsic_vectors:
            z = torch.tensor(vector,device=self.device,dtype=self.dtype)
            # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
            z = self.linear_V(z).reshape(self.n_prompt_tokens_V,-1)
            #z = z + self.position_V
            visual_prompt_list.append(z)

        return visual_prompt_list

    def metric(self,logits,label):
        ce_loss = F.cross_entropy(logits, label, reduction='none')
        final_loss = 0
        if self.loss_type == "ce":
            final_loss = torch.sum(ce_loss)
        elif self.loss_type == "focal":
            gamma = 2
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** gamma * ce_loss
            final_loss = torch.sum(focal_loss)
        return final_loss # Trả về tổng loss cho batch

    # Bỏ @torch.no_grad() khỏi eval nếu bạn muốn thực hiện adversarial training
    # Nếu chỉ muốn đánh giá robustness sau khi tối ưu, giữ nguyên @torch.no_grad() ở đây
    @torch.no_grad()
    def eval(self,prompt_zip):
        # ... (Phần còn lại của eval giữ nguyên) ...
        prompt_text,prompt_image = prompt_zip[0],prompt_zip[1]
        self.num_call += 1
        loss = 0
        if self.parallel:
            loss = [0]*self.popsize
        text_features = self.text_encoder(prompt_text) # if parallel, text_features.shape = [n_cls * popsize, *, *]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for batch in self.train_loader:
            image,label = self.parse_batch(batch)
            image_features = self.image_encoder(image,prompt_image)
            image_features = image_features / image_features.norm(dim=-1,keepdim=True)
            logit_scale = self.logit_scale.exp()
            if self.parallel:
                B = int(image_features.shape[0]/self.popsize)
                for i in range(self.popsize):
                    start_text = i * self.n_cls
                    start_image = i * B
                    tmp_text_features = text_features[start_text:start_text+self.n_cls]
                    tmp_image_features = image_features[start_image:start_image+B]
                    tmp_logits =  logit_scale*tmp_image_features@tmp_text_features.t()
                    # Sử dụng reduction='sum' trong metric hoặc sum ở đây
                    loss[i]+=self.metric(tmp_logits,label) # metric trả về sum loss
            else:
                logits = logit_scale*image_features@text_features.t()
                # Sử dụng reduction='sum' trong metric hoặc sum ở đây
                loss +=self.metric(logits,label) # metric trả về sum loss

        epoch_min_loss = None
        if self.parallel:
            loss = [x/len(self.train_data) for x in loss]
            epoch_min_loss = min(loss)
        else:
            loss /= len(self.train_data)
            epoch_min_loss = loss # Gán loss trực tiếp

        # Cập nhật min_loss và best_prompt
        if self.min_loss is None or epoch_min_loss < self.min_loss:
             self.min_loss = epoch_min_loss
             if self.parallel:
                 index = loss.index(epoch_min_loss)
                 # Lưu trữ bản sao sâu để tránh thay đổi không mong muốn
                 self.best_prompt_text = copy.deepcopy(prompt_text[index])
                 self.best_prompt_image = copy.deepcopy(prompt_image[index])
             else:
                 self.best_prompt_text = copy.deepcopy(prompt_text)
                 self.best_prompt_image = copy.deepcopy(prompt_image)


        if self.num_call % self.test_every == 0:
            # Gọi test để lấy cả clean và adv accuracy
            clean_acc, adv_acc = self.test(attack=True)
            self.acc.append(clean_acc.item()) # Lưu clean accuracy
            # Cập nhật best_accuracy dựa trên clean accuracy
            if clean_acc > self.best_accuracy:
                self.best_accuracy = clean_acc.item()
                print(f"*** New Best Clean Accuracy: {self.best_accuracy:.4f} ***")

            # In thêm adv_acc
            print(f"Adversarial Accuracy (PGD-{self.pgd_steps}, eps={self.pgd_eps:.4f}): {adv_acc:.4f}")

            #---------------save_results-----------------------------------
            output_dir = os.path.join(self.output_dir,self.task_name)

            fname = "{}_{}_{}.pth".format(self.task_name, self.opt_name, self.backbone.replace("/","-"))

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,
                       "best_clean_accuracy":self.best_accuracy, # Đổi tên cho rõ ràng
                       "clean_acc_history":self.acc, # Đổi tên cho rõ ràng
                       "best_prompt_text":self.best_prompt_text,
                       "best_prompt_image":self.best_prompt_image,
                       "loss_history":self.loss,"num_call":self.num_call, # Đổi tên cho rõ ràng
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict(),
                       # Có thể lưu thêm kết quả adversarial cuối cùng nếu muốn
                       "final_adv_acc": adv_acc.item() if adv_acc is not None else None
                       }
            Analysis_Util.save_results(content,output_dir,fname)
            # ---------------save_results-----------------------------------
        return loss

    # Bỏ @torch.no_grad() vì PGD cần tính gradient cho ảnh
    # @torch.no_grad() -> Xóa dòng này
    def test(self, attack=False):
        """
        Đánh giá mô hình trên tập test.
        Args:
            attack (bool): True để thực hiện tấn công PGD và trả về cả clean/adv accuracy.
                           False để chỉ tính clean accuracy.
        Returns:
            torch.Tensor: Clean accuracy.
            torch.Tensor or None: Adversarial accuracy nếu attack=True, ngược lại là None.
        """
        if self.best_prompt_text is None or self.best_prompt_image is None:
             print("Warning: best_prompt not found. Testing with potentially uninitialized prompts.")
             # Xử lý trường hợp chưa có best prompt (ví dụ: dùng prompt mặc định hoặc báo lỗi)
             # Tạm thời trả về 0 accuracy
             return torch.tensor(0.0), torch.tensor(0.0) if attack else None


        correct_clean = 0.
        correct_adv = 0.
        total = 0.

        # --- Quan trọng: Đặt lại parallel=False cho việc test ---
        original_parallel_state = self.parallel
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False
        # ------------------------------------------------------

        # Tính text features một lần vì nó không đổi trong quá trình test
        with torch.no_grad():
            text_features = self.text_encoder(self.best_prompt_text)
            text_features = text_features / text_features.norm(dim=-1,keepdim=True)

        for batch in self.test_loader:
            image_clean, label = self.parse_batch(batch) # Lấy ảnh gốc và nhãn
            label = label.to(self.device)
            image_clean = image_clean.to(self.device) # Đảm bảo ảnh trên đúng device
            total += label.size(0)

            # --- 1. Đánh giá Clean Accuracy ---
            with torch.no_grad():
                image_features_clean = self.image_encoder(image_clean, self.best_prompt_image)
                image_features_clean = image_features_clean / image_features_clean.norm(dim=-1,keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_clean = logit_scale * image_features_clean @ text_features.t()
                prediction_clean = logits_clean.argmax(dim=-1)
                correct_clean += (prediction_clean == label).float().sum()

            # --- 2. Tấn công PGD và Đánh giá Adversarial Accuracy (nếu attack=True) ---
            if attack:
                # Bật gradient cho ảnh đầu vào
                image_adv = image_clean.clone().detach().requires_grad_(True)
                # Lưu ảnh gốc để chiếu lại
                image_orig = image_clean.clone().detach()

                for _ in range(self.pgd_steps):
                    # Bật tính toán gradient trong ngữ cảnh này
                    with torch.enable_grad():
                        # Tính image features cho ảnh bị nhiễu
                        image_features_adv = self.image_encoder(image_adv, self.best_prompt_image)
                        image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)

                        # Tính logits và loss
                        logits_adv = logit_scale * image_features_adv @ text_features.t()
                        loss = F.cross_entropy(logits_adv, label)

                    # Tính gradient của loss theo ảnh bị nhiễu
                    grad = torch.autograd.grad(loss, image_adv,
                                               retain_graph=False, create_graph=False)[0]

                    # Cập nhật ảnh theo PGD step
                    image_adv = image_adv.detach() + self.pgd_alpha * grad.sign()
                    # Chiếu nhiễu vào L-infinity ball (epsilon)
                    delta = torch.clamp(image_adv - image_orig, min=-self.pgd_eps, max=self.pgd_eps)
                    # Đảm bảo ảnh + nhiễu nằm trong khoảng giá trị hợp lệ (sau normalization)
                    image_adv = torch.clamp(image_orig + delta, min=self.clip_min, max=self.clip_max).detach().requires_grad_(True)


                # Đánh giá trên ảnh đã bị tấn công (không cần gradient nữa)
                with torch.no_grad():
                    image_features_adv_final = self.image_encoder(image_adv, self.best_prompt_image)
                    image_features_adv_final = image_features_adv_final / image_features_adv_final.norm(dim=-1,keepdim=True)
                    logits_adv_final = logit_scale * image_features_adv_final @ text_features.t()
                    prediction_adv = logits_adv_final.argmax(dim=-1)
                    correct_adv += (prediction_adv == label).float().sum()

        # --- Khôi phục trạng thái parallel ---
        self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = original_parallel_state
        # ------------------------------------

        clean_acc = correct_clean / total
        adv_acc = correct_adv / total if attack else None

        #print(f"Clean Accuracy: {clean_acc:.4f}")
        #if attack:
        #    print(f"Adversarial Accuracy (PGD-{self.pgd_steps}, eps={self.pgd_eps:.4f}): {adv_acc:.4f}")

        return clean_acc, adv_acc # Trả về cả hai

    # ... (load_dataset và parse_batch giữ nguyên) ...
    def load_dataset(self):
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess)
            self.test_data, self.test_loader = load_test_cifar100(batch_size=self.batch_size, preprocess=self.preprocess)
        elif self.task_name == 'StanfordCars':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Cars_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'OxfordPets':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="OxfordPets_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'UCF-101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="UCF-101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'DTD':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="DTD_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'EuroSAT':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="EuroSAT_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'Food101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="Food101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'caltech101':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="caltech101_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'SUN397':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="SUN397_Gen")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)
        elif self.task_name == 'ImageNet':
            self.train_data,self.train_loader = load_train(batch_size=self.batch_size,seed=self.seed,shots=self.k_shot,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.test_data,self.test_loader = load_test(batch_size=self.batch_size,preprocess=self.preprocess,
                                                           root=self.data_dir,dataset_dir="imagenet")
            self.classes = self.train_data.classes
            self.n_cls = len(self.classes)

    def parse_batch(self,batch):
        image = batch["image"]
        label = batch["label"]
        # Chuyển đổi dtype và device ở đây thay vì trong test/eval để tránh lặp lại
        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)
        if self.parallel:
            # Lặp lại image cho parallel evaluation trong eval
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label
