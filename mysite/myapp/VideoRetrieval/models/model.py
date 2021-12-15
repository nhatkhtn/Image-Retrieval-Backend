import clip
import torch 
from torch import nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model, self.preprocess = clip.load("RN50x16")

    def encode_image(self, images):
        return self.model.encode_image(images)
        
    def encode_text(self, text):
        text = clip.tokenize(text).to(next(self.model.parameters())).int()
        return self.model.encode_text(text)
    
    def calc_similarity(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    