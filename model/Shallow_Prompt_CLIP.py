import os
import torch
from torch.nn import functional as F
import numpy as np
import clip
from torchattacks import PGD, TPGD
from torchvision.datasets import CIFAR100
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
from model.shallow_encoder import TextEncoder,VisionEncoder
from model.analysis_utils import Analysis_Util
from dataset.general import load_train,load_test
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
        return final_loss

    @torch.no_grad()
    def eval(self,prompt_zip):
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
                    loss[i]+=self.metric(tmp_logits,label)
            else:
                logits = logit_scale*image_features@text_features.t()
                loss +=self.metric(logits,label)

        epoch_min_loss = None
        if self.parallel:
            loss = [x/len(self.train_data) for x in loss]
            epoch_min_loss = min(loss)
        else:
            loss /= len(self.train_data)
            epoch_min_loss = loss if epoch_min_loss == None else min(loss,epoch_min_loss)
        self.loss.append(loss)

        if self.min_loss is None or epoch_min_loss<self.min_loss:
            self.min_loss = epoch_min_loss
            if self.parallel:
                index = loss.index(epoch_min_loss)
                self.best_prompt_text = prompt_text[index]
                self.best_prompt_image = prompt_image[index]
            else:
                self.best_prompt_text = prompt_text
                self.best_prompt_image = prompt_image

        #num_call = self.num_call*self.popsize if self.parallel else self.num_call


        if self.num_call % self.test_every == 0:
            acc = self.test()
            self.acc.append(acc)
            self.best_accuracy = max(acc,self.best_accuracy)
            #---------------save_results-----------------------------------
            output_dir = os.path.join(self.output_dir,self.task_name)

            fname = "{}_{}_{}.pth".format(self.task_name, self.opt_name, self.backbone.replace("/","-"))
            # fname = "{}_intrinsic_{}.pth".format(self.task_name, self.intrinsic_dim_L)

            content = {"task_name":self.task_name,"opt_name":self.opt_name,"backbone":self.backbone,"best_accuracy":self.best_accuracy,"acc":self.acc,
                       "best_prompt_text":self.best_prompt_text,"best_prompt_image":self.best_prompt_image,"loss":self.loss,"num_call":self.num_call,
                       "Linear_L":self.linear_L.state_dict(),"Linear_V":self.linear_V.state_dict()}
            Analysis_Util.save_results(content,output_dir,fname)
            # ---------------save_results-----------------------------------
            #print("current loss: {}".format(self.min_loss))
        return loss

    # @torch.no_grad()
    # import torch
# Giả sử PGD đã được import, ví dụ: from torchattacks import PGD
# Giả sử các biến model, eps, alpha, steps đã được định nghĩa ở đâu đó
# hoặc là thuộc tính của class (ví dụ: self.model, self.eps, ...)

def test(self):
    # --- Khởi tạo tấn công ---
    # Đảm bảo model, eps, alpha, steps có thể truy cập được
    # Nếu chúng là thuộc tính class, dùng self.model, self.eps,...
    # Ví dụ: attack = PGD(self.model, eps=self.eps, alpha=self.alpha, steps=self.steps)
    # Ở đây dùng biến cục bộ như trong code gốc của bạn:
    try:
        attack = PGD(self.model, eps=4/255, alpha=2.67/(4*255), steps=100)
    except NameError as e:
        print(f"Lỗi: Biến cho PGD chưa được định nghĩa ({e}). Đảm bảo 'model', 'eps', 'alpha', 'steps' có sẵn.")
        return None, None # Hoặc xử lý lỗi khác

    correct_clean = 0.
    correct_attacked = 0.
    total = 0

    # --- Lưu và tắt chế độ parallel nếu có ---
    parallel = self.parallel
    self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = False
    device = next(self.image_encoder.parameters()).device # Lấy device từ model

    # --- Tính text features một lần ---
    with torch.no_grad(): # Không cần gradient cho text features cố định
        text_features = self.text_encoder(self.best_prompt_text.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = self.logit_scale.exp()

    # --- Vòng lặp qua dữ liệu test ---
    for batch in self.test_loader:
        image, label = self.parse_batch(batch)
        image, label = image.to(device), label.to(device)
        batch_size = image.shape[0]
        total += batch_size

        # --- Đánh giá trên ảnh gốc (Clean Accuracy) ---
        with torch.no_grad(): # Không cần gradient khi đánh giá
            image_features_clean = self.image_encoder(image, self.best_prompt_image)
            image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
            logits_clean = logit_scale * image_features_clean @ text_features.t()
            prediction_clean = logits_clean.argmax(dim=-1)
            correct_clean += (prediction_clean == label).float().sum().item()

        # --- Tạo ảnh bị tấn công và đánh giá (Attacked Accuracy) ---
        # Bật lại gradient cho ảnh đầu vào để PGD hoạt động
        image.requires_grad = True
        attacked_image = attack(image, label) # Tạo ảnh bị tấn công
        image.requires_grad = False # Tắt lại gradient sau khi tấn công

        with torch.no_grad(): # Không cần gradient khi đánh giá ảnh bị tấn công
            image_features_attacked = self.image_encoder(attacked_image, self.best_prompt_image)
            image_features_attacked = image_features_attacked / image_features_attacked.norm(dim=-1, keepdim=True)
            logits_attacked = logit_scale * image_features_attacked @ text_features.t()
            prediction_attacked = logits_attacked.argmax(dim=-1)
            correct_attacked += (prediction_attacked == label).float().sum().item()

    # --- Khôi phục chế độ parallel ---
    self.parallel = self.text_encoder.parallel = self.image_encoder.parallel = parallel

    # --- Tính toán độ chính xác ---
    # Sử dụng total thay vì len(self.test_data) để chính xác hơn nếu batch cuối không đầy
    acc_clean = correct_clean / total if total > 0 else 0.
    acc_attacked = correct_attacked / total if total > 0 else 0.

    # In kết quả (tùy chọn)
    # print(f"Clean Accuracy: {acc_clean:.4f}")
    # print(f"Attacked Accuracy (PGD): {acc_attacked:.4f}")

    return acc_clean, acc_attacked

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
        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)
        if self.parallel:
            image = image.repeat(self.popsize, 1, 1, 1)
        return image, label
