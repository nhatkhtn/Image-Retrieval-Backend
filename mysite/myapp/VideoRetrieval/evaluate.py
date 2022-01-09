import torch
from tqdm import tqdm 
from torch.utils.data import DataLoader
import time 
# 
from models.model import Model 
from datasets.video import VideoDataset
from datasets.description import DescriptionDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)

videoDataset = VideoDataset(root = "../../../images/TRECVid Data/testing_set", num_imgs = 8, subset = 'test', preprocess = model.preprocess)
descDataset = DescriptionDataset(root = "../../../images/TRECVid Data/testing_set", subset = 'test')

videoDataloader = DataLoader(videoDataset, batch_size = 16, shuffle = False, num_workers = 8)
descDataloader = DataLoader(descDataset, batch_size = 16, shuffle = False, num_workers = 8)

def getVideoFeatures(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['images'].to(device)           # B, C, H, W 
            vid_ids = data['video_id']   # B 

            # IMAGES AS VIDEO 
            num_imgs = data['num_imgs'][0]
            # print(images.shape)
            images = images.flatten(start_dim = 0, end_dim = 1) 
            # print(images.shape)
            features = model.encode_image(images)

            all_features.append(features)
            all_labels.append(vid_ids.unsqueeze(-1).repeat(1, num_imgs).view(-1)) 

    return torch.cat(all_features), torch.cat(all_labels)

def getTextFeatures(dataloader):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            description = data['description']
            vid_ids = data["video_id"]
            features = model.encode_text(description)
            
            all_features.append(features)
            all_labels.append(vid_ids) 
    return torch.cat(all_features), torch.cat(all_labels)


process_begin = time.time()
# Calculate the image features
videoFeatures, videoLabels = getVideoFeatures(videoDataloader)
textFeatures, textLabel = getTextFeatures(descDataloader)
target = textLabel.view(-1, 1) == videoLabels.view(1, -1)

logits_per_image, logits_per_text = model.calc_similarity(videoFeatures, textFeatures)

total_process_time = time.time() - process_begin
print(f"Total process time: {total_process_time:03f}")

from torchmetrics.functional import *
mAP = torch.tensor([retrieval_average_precision(logits_per_text[i], target[i]) for i in range(logits_per_text.size(0))]).mean()
print(f"Mean Average Precision: {mAP}")
k = 15 
recK = torch.tensor([retrieval_recall(logits_per_text[i], target[i], k) for i in range(logits_per_text.size(0))]).mean()
print(f"Mean Recall@{k}: {recK}")
rr  = torch.tensor([retrieval_reciprocal_rank(logits_per_text[i], target[i]) for i in range(logits_per_text.size(0))]).mean()
print(f"Mean Reciprocal Rank: {rr}")
