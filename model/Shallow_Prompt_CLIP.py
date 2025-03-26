import torch
import argparse
import yaml
from tqdm import tqdm
from algorithm.CMA_ES import shallow_cma
from algorithm.LM_CMA_ES import Shallow_LMCMAES
from algorithm.MMES import Shallow_MMES
from algorithm.LMMAES import Shallow_LMMAES
from model.Shallow_Prompt_CLIP import PromptCLIP_Shallow
import numpy as np
import time
import os # Thêm os

__classification__ = ["CIFAR100","caltech101","StanfordCars","OxfordPets","UCF-101","DTD","EuroSAT",
                      "Food101","SUN397","ImageNet"]
__pypop__ = ["shallow_lmcmaes","shallow_mmes","shallow_dcem","shallow_maes"]
__dataset__ = "/home/yu/dataset"
__output__ = "/home/yu/dataset/result"
# __output__ = "/home/yu/result"
__backbone__ = "ViT-B/32"

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default="CIFAR100", type=str)
parser.add_argument("--opt", default="shallow_cma", type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
# Thêm tùy chọn để bật/tắt PGD test cuối cùng
parser.add_argument("--pgd_test", action='store_true', help='Perform PGD robustness test at the end.')


args = parser.parse_args()
assert "shallow" in args.opt, "Only shallow prompt tuning is supported in this file."
# --- Sửa đường dẫn YAML ---
config_path = os.path.join(os.path.dirname(__file__), "configs/shallow_prompt.yaml")
cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)
# -------------------------

cfg["opt_name"] = args.opt
cfg["data_dir"] = __dataset__
cfg["output_dir"] = __output__
cfg["opt_name"] = args.opt
cfg["backbone"] = __backbone__

# Cập nhật cfg với các giá trị cụ thể cho task và PGD (nếu có trong file yaml)
for k,v in cfg[args.task_name].items():
    cfg[k]=v
cfg["parallel"] = args.parallel

# Thêm các giá trị PGD mặc định nếu chưa có trong cfg
cfg.setdefault("pgd_eps", 8/255)
cfg.setdefault("pgd_alpha", 2/255)
cfg.setdefault("pgd_steps", 10)

device = "cuda" if torch.cuda.is_available() else "cpu"
intrinsic_dim_L = cfg["intrinsic_dim_L"]
intrinsic_dim_V = cfg["intrinsic_dim_V"]

# Eval function and Settings(if needed)+
# Hàm fitness_eval sẽ được gọi bởi các thuật toán tối ưu
# Nó cần trả về loss (hoặc giá trị fitness) để tối ưu hóa
def fitness_eval(prompt_zip):
    # Generate prompts from the intrinsic vector 'prompt_zip'
    # prompt_zip là một vector [intrinsic_dim_L + intrinsic_dim_V]
    prompt_text_list = prompt_clip.generate_text_prompts([prompt_zip[:intrinsic_dim_L]]) # Chỉ một individual
    prompt_image_list = prompt_clip.generate_visual_prompts([prompt_zip[intrinsic_dim_L:]]) # Chỉ một individual

    # prompt_clip.eval xử lý một cặp (text_prompt, image_prompt)
    # Nó tính loss trên tập train và thực hiện test định kỳ
    # Trả về loss trung bình trên tập train cho cặp prompt này
    fitness = prompt_clip.eval( (prompt_text_list[0], prompt_image_list[0]) ) # Truyền tuple

    # In thông tin (có thể đã được xử lý bên trong prompt_clip.eval)
    # if prompt_clip.num_call % (prompt_clip.test_every) == 0:
    #     print("-------------------------Epoch {}---------------------------".format(prompt_clip.num_call/prompt_clip.test_every))
    # if prompt_clip.num_call % (prompt_clip.popsize) == 0:
    #     print("Evaluation of Individual: {}, Generation: {}".format(prompt_clip.num_call % prompt_clip.popsize,
    #                                                             int(prompt_clip.num_call / prompt_clip.popsize)))
    # if prompt_clip.num_call % prompt_clip.test_every == 0:
    #     print("current loss: {}".format(prompt_clip.min_loss))
    #     print("Best Prompt Embedding - Acc : " + str(prompt_clip.best_accuracy))

    # Các thuật toán tối ưu thường tối thiểu hóa, nên trả về loss
    return fitness # fitness ở đây là loss


ndim_problem = intrinsic_dim_L + intrinsic_dim_V

# --- Cấu hình cho các thuật toán tối ưu ---
# Cấu hình chung (sử dụng bởi pypop)
pro = {'fitness_function': fitness_eval,
       'ndim_problem': ndim_problem}
opt_cfg = {'fitness_threshold': 1e-10, # Ngưỡng dừng sớm (ít dùng với max_runtime)
           'seed_rng': 0,
           'max_runtime': 20800, # Thời gian chạy tối đa (giây)
           'x': 0 * np.ones((ndim_problem,)),  # Điểm bắt đầu (mean)
           'sigma': cfg['sigma'], # Độ lệch chuẩn ban đầu
           'verbose_frequency': 5, # Tần suất in log
           'n_individuals': cfg["popsize"], # Kích thước quần thể
           'is_restart': False} # Có khởi động lại không

# Cấu hình riêng cho shallow_cma (nếu cần)
cma_cfg = cfg.copy() # Tạo bản sao để tránh thay đổi cfg gốc
cma_cfg.update(opt_cfg) # Gộp cấu hình chung vào

# ------------------------------------------

# Load algorithm
opt = None
if args.opt == "shallow_cma":
    # shallow_cma có thể có cách nhận config khác, kiểm tra implementation của nó
    # Giả sử nó nhận trực tiếp dict cfg
    opt = shallow_cma(cma_cfg) # Truyền dict config đã gộp
elif args.opt == "shallow_lmcmaes":
    opt = Shallow_LMCMAES(pro, opt_cfg)
elif args.opt == "shallow_mmes":
    opt = Shallow_MMES(pro, opt_cfg)
elif args.opt == "shallow_lmmaes":
    opt = Shallow_LMMAES(pro,opt_cfg)
# Thêm các thuật toán khác nếu có
# elif args.opt == "shallow_dcem":
#     opt = Shallow_DCEM(pro, opt_cfg) # Ví dụ
# elif args.opt == "shallow_maes":
#     opt = Shallow_MAES(pro, opt_cfg) # Ví dụ


# Build CLIP model
prompt_clip = PromptCLIP_Shallow(args.task_name,cfg) # Truyền cfg vào PromptCLIP_Shallow

# --- Phần khởi tạo ban đầu (có thể không cần thiết nếu thuật toán tự xử lý) ---
# text_context = prompt_clip.get_text_information()
# image_context = prompt_clip.get_image_information()
# prompt_clip.text_encoder.set_context(text_context)
# prompt_clip.image_encoder.set_context(image_context)
# solutions = opt.ask() # Lấy giải pháp ban đầu từ thuật toán
# prompt_text_list= prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
# prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])
# if cfg["parallel"]:
#      # Đánh giá song song nếu được hỗ trợ
#      initial_fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list])
#      initial_fitnesses = [f.item() for f in initial_fitnesses]
# else:
#      # Đánh giá tuần tự
#      initial_fitnesses = [prompt_clip.eval(x).item() for x in zip(prompt_text_list, prompt_image_list)]
# print("Initial Population Evaluation Done.")
# print("Original Acc (based on initial best): " + str(prompt_clip.test(attack=False)[0].item())) # Chỉ lấy clean acc
# -----------------------------------------------------------------------------


print('Population Size: {}'.format(cfg["popsize"]))
print(f'Using Optimizer: {args.opt}')
print(f'Task: {args.task_name}')
print(f'Backbone: {cfg["backbone"]}')
print(f'Parallel Evaluation: {cfg["parallel"]}')

# Black-box prompt tuning
start_time = time.time()

if args.opt in __pypop__:
    # Các thuật toán từ thư viện pypop thường có phương thức optimize()
    # Chúng sẽ tự gọi fitness_function (fitness_eval của chúng ta)
    print(f"Starting optimization with {args.opt} using pypop interface...")
    res = opt.optimize() # optimize() sẽ chạy vòng lặp ask-tell bên trong
    print("Optimization finished.")
    # Kết quả có thể nằm trong 'res' hoặc trong prompt_clip (do fitness_eval cập nhật)

else:
    # Các thuật toán khác (như shallow_cma tự viết) có thể dùng vòng lặp ask-tell rõ ràng
    print(f"Starting optimization with {args.opt} using ask-tell loop...")
    generation = 0
    while not opt.stop(): # Kiểm tra điều kiện dừng của thuật toán
        generation += 1
        print(f"\n--- Generation {generation} ---")
        solutions = opt.ask() # Lấy tập giải pháp mới từ thuật toán

        # Generate prompts for the current solutions
        prompt_text_list= prompt_clip.generate_text_prompts([x[:intrinsic_dim_L] for x in solutions])
        prompt_image_list = prompt_clip.generate_visual_prompts([x[intrinsic_dim_L:] for x in solutions])

        # Evaluate the solutions
        if cfg["parallel"]:
            # Đánh giá song song (prompt_clip.eval xử lý việc này)
            print("Evaluating solutions in parallel...")
            fitnesses = prompt_clip.eval([prompt_text_list, prompt_image_list]) # eval trả về list loss
            fitnesses = [x.item() for x in tqdm(fitnesses,ncols=80, desc="Parallel Eval")]
        else:
            # Đánh giá tuần tự
            print("Evaluating solutions sequentially...")
            fitnesses = []
            # Sử dụng zip để kết hợp text và image prompts cho từng solution
            for i, p_pair in enumerate(tqdm(zip(prompt_text_list, prompt_image_list), total=len(solutions), ncols=80, desc="Sequential Eval")):
                 # Gọi eval cho từng cặp prompt (text, image)
                 # Lưu ý: eval bây giờ thực hiện test định kỳ bên trong
                 loss = prompt_clip.eval(p_pair)
                 fitnesses.append(loss.item()) # Lấy giá trị loss

        # Cung cấp kết quả đánh giá lại cho thuật toán
        opt.tell(solutions, fitnesses)

        # In thông tin (có thể đã được in bên trong prompt_clip.eval khi test được thực hiện)
        # if prompt_clip.num_call % prompt_clip.test_every == 0:
        #     print(f"Current Min Loss: {prompt_clip.min_loss:.6f}")
        #     print(f"Current Best Clean Acc: {prompt_clip.best_accuracy:.4f}")

    print("Optimization finished.")


end_time = time.time()
print(f"Total optimization time: {end_time - start_time:.2f} seconds")

# --- Đánh giá cuối cùng sau khi tối ưu xong ---
print("\n--- Final Evaluation ---")
# Gọi test(attack=True) để lấy cả clean và adversarial accuracy cuối cùng
final_clean_acc, final_adv_acc = prompt_clip.test(attack=args.pgd_test)

print(f"Final Best Clean Accuracy: {final_clean_acc.item():.4f}")
if args.pgd_test and final_adv_acc is not None:
    print(f"Final Adversarial Accuracy (PGD-{prompt_clip.pgd_steps}, eps={prompt_clip.pgd_eps:.4f}): {final_adv_acc.item():.4f}")
elif args.pgd_test:
    print("Adversarial test requested but result is None.")

# Lưu lại kết quả cuối cùng một lần nữa (cập nhật final_adv_acc)
output_dir = os.path.join(prompt_clip.output_dir, prompt_clip.task_name)
fname = "{}_{}_{}.pth".format(prompt_clip.task_name, prompt_clip.opt_name, prompt_clip.backbone.replace("/","-"))
content = {"task_name":prompt_clip.task_name,"opt_name":prompt_clip.opt_name,"backbone":prompt_clip.backbone,
           "best_clean_accuracy":prompt_clip.best_accuracy, # Best clean acc trong quá trình train
           "final_clean_accuracy": final_clean_acc.item(), # Clean acc cuối cùng
           "clean_acc_history":prompt_clip.acc,
           "best_prompt_text":prompt_clip.best_prompt_text,
           "best_prompt_image":prompt_clip.best_prompt_image,
           "loss_history":prompt_clip.loss,"num_call":prompt_clip.num_call,
           "Linear_L":prompt_clip.linear_L.state_dict(),"Linear_V":prompt_clip.linear_V.state_dict(),
           "final_adv_acc": final_adv_acc.item() if final_adv_acc is not None else None, # Adv acc cuối cùng
           "pgd_eps": prompt_clip.pgd_eps,
           "pgd_alpha": prompt_clip.pgd_alpha,
           "pgd_steps": prompt_clip.pgd_steps
           }
Analysis_Util.save_results(content,output_dir,fname)
print(f"Final results saved to {os.path.join(output_dir, fname)}")

# pass # Không cần thiết
