from pathlib import Path
import os 
import glob 

import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm 

# IMAGE_ROOT = Path("/home/nero/Courses/CS412/Image-Retrieval-Backend/images")
IMAGE_ROOT = Path("/home/nero/Courses/CS412/Image-Retrieval-Backend/images/testing_set")
model, preprocess = (None, None)
imageFeatures = None 

def query_by_caption(caption):
    device = "cuda"

    global model, preprocess, imageFeatures

    if model is None:
        model, preprocess = clip.load("ViT-B/16", device=device)

    text = clip.tokenize([caption]).to(device)

    # image_paths = image_paths[:500] 
    image_paths = list(Path(IMAGE_ROOT).rglob('*.jpeg'))
    if imageFeatures is None:
        print("Preprocess images feature")
        print(len(image_paths))
        images = list(map(lambda img_path : preprocess(Image.open(img_path)), image_paths))
        images = torch.tensor(np.stack(images)).cuda()
        imageFeatures = [] 
        with torch.no_grad():
            for i in tqdm(range(images.size(0))):
                features = model.encode_image(images[i:i+1])
                imageFeatures.append(features)
            imageFeatures = torch.cat(imageFeatures, dim = 0)
            imageFeatures = imageFeatures / imageFeatures.norm(dim=-1, keepdim=True)
        
            
    with torch.no_grad():
        textFeatures = model.encode_text(text)
        textFeatures = textFeatures / textFeatures.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * imageFeatures @ textFeatures.t()
        logits_per_text = logits_per_image.t()
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        pairs = list(zip(image_paths, probs[0]))

    sorted_paths, sorted_probs = zip(*sorted(pairs, key=lambda x: x[1], reverse=True))
    # sorted_names = [str(img_path).replace(str(IMAGE_ROOT) + '/', "") for img_path in sorted_paths]
    sorted_names = [str(img_path).replace('/', '*') for img_path in sorted_paths]
    # print(sorted_names)
    # sorted_names = sorted_paths
    # print(sorted_names, sorted_probs)
    return sorted_names