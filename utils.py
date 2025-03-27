import os
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torchvision.datasets import CIFAR100
from dataset.cifar100 import load_train_cifar100, load_test_cifar100
# from model.shallow_encoder import TextEncoder,VisionEncoder
from dataset.general import load_train,load_test
from clip import clip
# from trainers.apt import PromptLearner, TextEncoder
from utils import *
from addict import Dict
from torchattacks import PGD

mu = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


class CustomCLIP(nn.Module):
    def __init__(self,
                 model,
                 preprocess,
                 task_name,
                 cls_prompt='a photo of a {}',
                 atk_prompt=None,
                 cfg=None):
        super().__init__()
        self.task_name = task_name
        self.preprocess = preprocess
        self.opt_name = cfg["opt_name"]
        self.data_dir = cfg["data_dir"]
        self.output_dir = cfg["output_dir"]
        self.backbone = cfg["backbone"]
        self.popsize = cfg["popsize"]
        self.parallel = cfg["parallel"]
        self.batch_size = cfg["batch_size"]
        self.k_shot = cfg["k_shot"]
        self.seed = cfg["seed"]
        self.logit_scale = model.logit_scale
        self.load_dataset()
        self.classnames = [name.replace("_", " ").replace("-"," ") for name in self.classes]
        self.model = model
        self.mode = 'classification'
        self.normalize = ImageNormalizer(mu, std).cuda()
        
        self.set_prompts(cls_prompt, atk_prompt)
        
    def _prompt_text_features(self, prompt):
        if '{}' in prompt:
            # manual prompt template
            prompts = torch.cat([clip.tokenize(prompt.format(c))
                                 for c in self.classnames])
            # self.model = self.model
            text_features = self.model.encode_text(prompts)
        # else:
        #     # optimized prompt vector
        #     prompter_ckp = prompt
        #     assert os.path.isfile(prompter_ckp)
        #     prompter = PromptLearner(self.cfg, self.classnames, self.model)
            
        #     state_dict = torch.load(prompter_ckp)["state_dict"]

        #     # Ignore fixed token vectors
        #     if "token_prefix" in state_dict:
        #         del state_dict["token_prefix"]

        #     if "token_suffix" in state_dict:
        #         del state_dict["token_suffix"]

        #     prompter.load_state_dict(state_dict, strict=False)
        #     text_encoder = TextEncoder(self.model)
        #     prompts = prompter()
        #     text_features = text_encoder(prompts, prompter.tokenized_prompts)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()
        
    def set_prompts(self, cls_prompt, atk_prompt=None):
        print(f'classification prompt: {cls_prompt}')
        self.cls_tfeatures = self._prompt_text_features(cls_prompt).cuda()
        
        if atk_prompt is None or cls_prompt == atk_prompt:
            print(f'attack prompt: {cls_prompt}')
            self.atk_tfeatures = self.cls_tfeatures
        else:
            print(f'attack prompt: {atk_prompt}')
            self.atk_tfeatures = self._prompt_text_features(atk_prompt).cuda()
            
    def forward(self, image):
        image_features = self.model.encode_image(self.normalize(image))        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        text_features = self.cls_tfeatures if self.mode == 'classification' else self.atk_tfeatures
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    def test(self):
        meters = Dict()
        meters.acc = AverageMeter('Clean Acc@1', ':6.2f')
        meters.rob = AverageMeter('Robust Acc@1', ':6.2f')
        
        progress = ProgressMeter(
            len(self.test_loader),
            [meters.acc, meters.rob],
            prefix=self.task_name)

        eps = 4/255
        alpha = eps / 4.0
        steps = 100
        
        
        attack = PGD(self, eps=eps, alpha=alpha, steps=steps)
        
            
        for i, data in enumerate(self.test_loader, start=1):
            
            imgs,tgts = data['image'],data['label']
            imgs, tgts = imgs.cuda(), tgts.cuda()
            bs = imgs.size(0)

            with torch.no_grad():
                output = self(imgs)

            acc = accuracy(output, tgts)
            meters.acc.update(acc[0].item(), bs)

            self.mode = 'attack'
            # if self.attack == 'aa':
            #     adv = attack.run_standard_evaluation(imgs, tgts, bs=bs)
            # elif args.attack in ['pgd', 'tpgd']:
            adv = attack(imgs, tgts)
            # else:
            #     adv, _ = pgd(imgs, tgts, model, CWLoss, eps, alpha, steps)
                
            self.mode = 'classification'

            # Calculate features
            with torch.no_grad():
                output = self(adv)

            rob = accuracy(output, tgts)
            meters.rob.update(rob[0].item(), bs)

            if i == 1 or i % 10 == 0 or i == len(self.test_loader):
                progress.display(i)
    def load_dataset(self):
        if self.task_name == 'CIFAR100':
            self.dataset = CIFAR100(self.data_dir, transform=self.preprocess, download=True)
            self.classes = self.dataset.classes
            self.n_cls = len(self.classes)
            # self.train_data,self.train_loader = load_train_cifar100(batch_size=self.batch_size,shots=self.k_shot,preprocess=self.preprocess)
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
    # def parse_batch(self,batch):
    #     image = batch["image"]
    #     label = batch["label"]
    #     image = image.to(device='cuda')
    #     label = label.to(device='cuda')
    #     # if self.parallel:
    #     #     image = image.repeat(self.popsize, 1, 1, 1)
    #     return image, label

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res