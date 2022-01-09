import clip
import torch 
from torch import nn 
from torch.nn import functional as F 

class Model(nn.Module):
    def __init__(self, cfg = None):
        super().__init__()
        self.cfg = cfg 
        self.model, self.preprocess = clip.load("RN101")

        # num_fc = cfg.num_fc
        # fc_dim_in = cfg.fc_dim_in 
        # fc_dim = cfg.fc_dim

        # self.fc_layers = []
        # for k in range(num_fc):
        #     fc = nn.Linear(fc_dim_in, fc_dim)
        #     # self.add_module("fc{}".format(k + 1), fc)
        #     self.fc_layers.append(fc)
        #     fc_dim_in = fc_dim
            
        # self.predictor = nn.Linear(fc_dim_in, 1)

    def encode_image(self, images):
        return self.model.encode_image(images)
        
    def encode_text(self, text):
        text = clip.tokenize(text).to(next(self.model.parameters())).int()
        return self.model.encode_text(text)
    
    # def forward(self, image_features, text_features):
    #     x = torch.cat([image_features, text_features], dim = -1)
    #     for layer in self.fc_layers:
    #         x = F.relu(layer(x))
    #     return torch.sigmoid(self.predictor(x))

    def calc_similarity(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.model.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t() # 
        logits_per_image = image_features @ text_features.t() # 
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    