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
import clip 

class VideoWithDescDataset(Dataset):
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
        
        # Video
        self.videos = []
        self.frames = {}
        
        vid_list = sorted(os.listdir(self.vid_root))
                 
        self.video_urls_path = path.join(root, f'{subset}_video_urls.csv')    
        video_urls = pd.read_csv(self.video_urls_path) 

        for i, row in video_urls.iterrows():
            vid_id = str(row['video_id']).zfill(5)
            if vid_id not in vid_list:
                continue  
            # frames = sorted(os.listdir(path.join(self.vid_root, vid_id)), key=lambda x: int(x.split('_')[1].split('.')[0]))
            frames = sorted(os.listdir(path.join(self.vid_root, vid_id)))
                
            if len(frames) < self.num_imgs:
                continue 
            
            self.frames[vid_id] = frames 
            self.videos.append(vid_id)

        self.videos.sort() 
        print(self.videos[:40])
        # Description
        self.short_text_descriptions = path.join(root, f'{subset}_text_descriptions.csv')  
        desc_csv = pd.read_csv(self.short_text_descriptions) 
        self.description = {}
        for i, row in desc_csv.iterrows():
            vid_id = str(row['video_id']).zfill(5)
            try:
                curr_description = row['description'] 
                if vid_id not in self.description:
                    self.description[vid_id] = []
                self.description[vid_id].append(curr_description)
            except:
                pass 
            for i in range(10):
                try:
                    curr_description = row[f'description_{i}'] 
                    if vid_id not in self.description:
                        self.description[vid_id] = []
                    self.description[vid_id].append(curr_description)
                except:
                    pass 
                    
        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), self.vid_root))
        
    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['video_id'] = video
        
        vid_im_path = path.join(self.vid_root, video)
        frames = self.frames[video]
        
        info['frames'] = []# Appended with actual frames
        
        if self.num_imgs == -1:
            frames_idx = frames
        else:
            # print(frames)
            # base = np.random.randint(0, len(frames) - 1)
            # frames_idx = [frames[max(0, base - self.num_imgs + i + 1)] for i in range(self.num_imgs)]
            frames_idx = sorted(np.random.choice(frames, size = self.num_imgs, replace = False))
            # print(frames_idx)
        images = []
        
        for f_idx in frames_idx:
            jpg_name = f_idx
            info['frames'].append(jpg_name)

            this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
            if self.preprocess:
                this_im = self.preprocess(this_im)
            # else:
            #     this_im = self.final_im_transform(this_im)
            images.append(this_im)

        images = torch.stack(images, 0)

        desc = random.choice(self.description[video])
        if len(desc) > 50:
            desc = ' '.join(desc.split()[:50])
        desc = clip.tokenize(desc)[0]    

        out = {
            'images': images,
            'num_imgs': self.num_imgs if self.num_imgs != -1 else len(frames_idx),
            'video_id': int(video),
            'description': desc,
            'info': info
        }
        return out

    def __len__(self):
        # return 4
        return len(self.videos)