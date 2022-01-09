import os
from os import path
import pandas as pd 

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from tqdm import tqdm
import random 

class VideoDataset(Dataset):
    """
    Works for Memento10k/TRECVid training
    For each sequence:
    - Pick {num_imgs} frames
    Option:
    # - Apply some random transforms that are the same for all frames
    # - Apply random transform to each of the frame
    # - The distance between frames is controlled
    """

    def __init__(self, name = 'trecvid', root = None, num_imgs = 4, subset = 'train', preprocess = None):
        self.root = root
        self.vid_root = path.join(root, 'Frames') 
        self.subset = subset 
        self.num_imgs = num_imgs
        self.preprocess = preprocess
        
        self.videos = []
        self.frames = {}
        
        vid_list = sorted(os.listdir(self.vid_root))
                 
        self.video_urls_path = path.join(root, f'{subset}_video_urls.csv')    
        video_urls = pd.read_csv(self.video_urls_path) 

        for i, row in video_urls.iterrows():
            vid_id = str(row['video_id']).zfill(5)
            if vid_id not in vid_list:
                continue  
            frames = sorted(os.listdir(path.join(self.vid_root, vid_id)))
                
            if len(frames) < self.num_imgs:
                continue 
            
            self.frames[vid_id] = frames 
            self.videos.append(vid_id)
    
        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), self.vid_root))
        im_mean = (124, 116, 104)
        # Final transform without randomness
        if subset == 'train':
            self.final_im_transform = transforms.Compose([
                transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
                transforms.Resize((320, 320), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            self.final_im_transform = transforms.Compose([
                # transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
                transforms.Resize((320, 320), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['video_id'] = video

        vid_im_path = path.join(self.vid_root, video)
        frames = self.frames[video]
        
        # if np.random.rand() < 0.5:
        #     # Reverse time
        #     frames = frames[::-1]
        
        info['frames'] = []# Appended with actual frames
        
        if self.num_imgs == -1:
            frames_idx = frames
        else:
            frames_idx = sorted(np.random.choice(frames, size = self.num_imgs, replace = False))
        
        images = []
        
        for f_idx in frames_idx:
            jpg_name = f_idx
            info['frames'].append(jpg_name)

            this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
            # this_im = self.final_im_transform(this_im)
            if self.preprocess:
                this_im = self.preprocess(this_im)
                # print(this_im.shape)
            else:
                this_im = self.final_im_transform(this_im)
            images.append(this_im)

        images = torch.stack(images, 0)

        out = {
            'images': images,
            'num_imgs': self.num_imgs if self.num_imgs != -1 else len(frames_idx),
            'video_id': int(video)
            # 'info': info
        }
        return out

    def __len__(self):
        return len(self.videos)