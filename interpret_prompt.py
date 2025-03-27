import os
import argparse
import torch
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip

def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="./")
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

parser = argparse.ArgumentParser()
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk
assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location='cpu')
ctx = prompt_learner['best_prompt_text']
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]
    
    formatted_lines = []
    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m+1}: {words} {dist}")
        formatted_lines.append(f"{m+1}: {' '.join(words)}")
    
    # In thêm 5 dòng liền mạch nhau
    print("\n".join(formatted_lines[:5]))

elif ctx.dim() == 3:
    raise NotImplementedError
