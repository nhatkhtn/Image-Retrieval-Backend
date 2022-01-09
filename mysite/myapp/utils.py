from pathlib import Path
import os 
import glob 

import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm 

from myapp.VideoRetrieval.datasets.video_with_desc import VideoWithDescDataset
from myapp.VideoRetrieval.network import Model 

IMAGE_ROOT = Path("/home/nero/Courses/CS412/Image-Retrieval-Backend/images/TRECVid Data/testing_set")
model, preprocess = (None, None)
imageFeatures = None 

device = "cuda"
used_temporalfusion = True 

def getVideoFeatures(model, dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['images'].to(device)           # B, T, C, H, W 
            vid_ids = data['video_id']   # B 

            # IMAGES AS VIDEO 
            num_imgs = data['num_imgs'][0]

            if used_temporalfusion: 
                images = images.unsqueeze(2).repeat(1, 1, 3, 1, 1, 1)
                images[:, 2:, 0] = images[:, :-2, 2]
                images[:, 1:, 0] = images[:, :-1, 2]
            else:
                images = images.unsqueeze(2).repeat(1, 1, 1, 1, 1, 1)

            images = images.flatten(start_dim = 0, end_dim = 1) 
            features = model.encode_images(images)

            all_features.append(features)
            all_labels.append(vid_ids.unsqueeze(-1).repeat(1, num_imgs).view(-1)) 

    return torch.cat(all_features), torch.cat(all_labels)

import torch
from tqdm import tqdm 
from torch.utils.data import DataLoader

def query_by_caption(caption):
    
    global model, preprocess, imageFeatures

    if model is None:
        model = Model().to(device)
    
    text = clip.tokenize([caption]).to(device)
    image_paths = sorted(list(Path(IMAGE_ROOT).rglob('*.jpeg')))
    # print(image_paths[:40])
    if imageFeatures is None:
        print("Preprocess images feature")
        
        videoDataset = VideoWithDescDataset(root = IMAGE_ROOT, num_imgs = 8, subset = 'test', preprocess = model.preprocess)
        videoDataloader = DataLoader(videoDataset, batch_size = 8, shuffle = False, num_workers = 8)
        videoFeatures, videoLabels = getVideoFeatures(model, videoDataloader)
        imageFeatures = videoFeatures / videoFeatures.norm(dim=-1, keepdim=True)
        
            
    with torch.no_grad():
        textFeatures = model.encode_text(text)
        textFeatures = textFeatures / textFeatures.norm(dim=-1, keepdim=True)

        logits_per_image = imageFeatures @ textFeatures.t()
        logits_per_text = logits_per_image.t()
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        pairs = list(zip(image_paths, probs[0]))

    sorted_paths, sorted_probs = zip(*sorted(pairs, key=lambda x: x[1], reverse=True))
    sorted_names = [str(img_path).replace('/', '*') for img_path in sorted_paths]
    return sorted_names