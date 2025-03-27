import torch.nn as nn
import clip
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

class ClipCustom(nn.Module):
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = clip_imsize
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to('cuda')
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

def get_text_information(model, classes, caption=None, device="cuda", dtype=torch.float32):
        # classification task - caption - None 
        # refcoco ask - caption - str
        prompt_prefix = " ".join(["X"] * 8)
        if caption is None:
            classnames = [name.replace("_", " ").replace("-"," ") for name in classes]
            pattern_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_pattern_prompts= torch.cat([clip.tokenize(p) for p in pattern_prompts]).to(device)
            with torch.no_grad():
                init_pattern_embedding = model.token_embedding(tokenized_pattern_prompts).type(dtype)
            
        else:
            pattern_prompt = prompt_prefix + caption + "."
            tokenized_pattern_prompts = torch.cat([clip.tokenize(pattern_prompt)]).to(device)
            with torch.no_grad():
                init_pattern_embedding = model.token_embedding(tokenized_pattern_prompts).type(dtype)
    
        return init_pattern_embedding