from pathlib import Path

import clip
import torch
import numpy as np
from PIL import Image

IMAGE_ROOT = Path("H:\Workspace\clip\images")

model, preprocess = (None, None)

def query_by_caption(caption):
    device = "cuda"

    global model, preprocess

    if model is None:
        model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize([caption]).to(device)

    image_paths = list(Path(IMAGE_ROOT).iterdir())
    images = list(map(lambda img_path : preprocess(Image.open(img_path)), image_paths))
    images = torch.tensor(np.stack(images)).cuda()

    print(images.shape)

    with torch.no_grad():
        # images_features = model.encode_image(images)
        # text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        pairs = list(zip(image_paths, probs[0]))
        print(pairs)

    sorted_paths, sorted_probs = zip(*sorted(pairs, key=lambda x: x[1], reverse=True))
    sorted_names = [img_path.name for img_path in sorted_paths]
    print(sorted_names, sorted_probs)
    return sorted_names