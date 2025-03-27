import torch.nn as nn

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