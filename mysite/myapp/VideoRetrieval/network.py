import clip
import torch 
from torch import nn 
import numpy as np 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/16")

    def encode_images(self, images):
        B, T = images.shape[:2]
        images = images.flatten(end_dim = 1)
        features = self.encode_image(images)
        features = features.reshape(B, T, *features.shape[1:])
        features = torch.mean(features, dim = 1)
        return features

    def encode_image(self, image):
        return self.model.encode_image(image)
        
    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def forward(self, images, text):
        visual_features = self.encode_images(images)
        textual_features = self.encode_text(text)

        out = torch.cat([visual_features, textual_features], dim = 0)
        return out 
    
    def calc_similarity(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t() # 
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
        
# from cbam import CBAM

# class CollaborativeMemory(nn.Module):
#     def __init__(self):
#         super().__init__() 
#         dim = 2048
#         key_dim = 2048 
#         self.linear_q = nn.Linear(dim, key_dim)
#         self.linear_k = nn.Linear(dim, key_dim)
#         self.linear_v = nn.Linear(dim, dim)
#         self.linear_o = nn.Linear(dim, dim)

#     def forward(self, cur, mem):
#         """
#             cur: B, C, H, W 
#             mem: B, T, C, H, W
#         """
        
#         B, T, C, H, W = mem.shape 
#         memp = mem.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
#         curp = cur.permute(0, 2, 3, 1).reshape(B, -1, C)
#         mem_k = self.linear_k(memp) # B, THW, Ck
#         mem_v = self.linear_v(memp.view(B, -1, C)) # B, THW, Cv
#         mem_push = mem_k.transpose(1, 2) @ mem_v # B, Ck, Cv 
#         cur_q = self.linear_q(curp) # B, HW, Ck 
#         mem_pop = (cur_q @ mem_push).transpose(1, 2).view(B, C, H, W) # B, HW, Cv
#         mem_pop = nn.AdaptiveAvgPool2d(1)(mem_pop).view(B, C)
#         weights = torch.sigmoid(self.linear_o(mem_pop))
#         out = cur  * weights.view(B, C, 1, 1)
#         # print(weights.min(), weights.max())
#         return out.view(B, C, H, W)

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model, self.preprocess = clip.load("ViT-B/16")

#         # self.attnpool = self.model.visual.attnpool
#         # self.model.visual.attnpool = nn.AvgPool2d(1, 1, 0)# torch.nn.Conv2d(2048, 2048, kernel_size = 1, padding = 0, stride = 1, bias = 0)
#         # self.memory = CollaborativeMemory() 
#         # self.fc = nn.Conv1d(3, 1, kernel_size=5,stride=1,padding=2, bias=False)
#         # self.cbam = CBAM(self.attnpool.c_proj.out_features, no_spatial=True)

#         # torch.manual_seed(42)
#         # torch.nn.init.normal_(self.fc.weight, 0, 0.5)
#         # self.fc.bias.data.fill_(0.01)
#         # print(self.fc.weight )
#     def encode_images(self, images):
#         B, T = images.shape[:2]
#         images = images.flatten(end_dim = 1)
#         features = self.encode_image(images)
#         features = features.reshape(B, T, *features.shape[1:])
#         # features = self.memory(features[:, -1], features[:, :-1])    
#         # features = self.attnpool(features)
#         # print(features.shape)
#         # features = features.reshape(B, T, *features.shape[1:]).permute(0, 2, 1).unsqueeze(-1)
#         # features = self.cbam(features).squeeze(-1).permute(0, 2, 1)
#         # features = self.fc(features).squeeze(1)
#         features = torch.mean(features, dim = 1)
#         return features

#     def encode_image(self, image):
#         return self.model.encode_image(image)
        
#     def encode_text(self, text):
#         return self.model.encode_text(text)
    
#     def forward(self, images, text):
#         visual_features = self.encode_images(images)
#         textual_features = self.encode_text(text)

#         out = torch.cat([visual_features, textual_features], dim = 0)
#         # logits_per_image, logits_per_text = self.calc_similarity(visual_features, textual_features)
#         # print(np.round(logits_per_image.detach().cpu().numpy(), 3))
#         # logits_per_image, logits_per_text = self.calc_similarity(out, out)
#         # print(np.round(logits_per_image.detach().cpu().numpy(), 3))
#         return out 
    
#     def calc_similarity(self, image_features, text_features):
#         # normalized features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # cosine similarity as logits
#         # logit_scale = self.model.logit_scale.exp()
#         # logits_per_image = logit_scale * image_features @ text_features.t() # 
#         logits_per_image = image_features @ text_features.t() # 
#         logits_per_text = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text