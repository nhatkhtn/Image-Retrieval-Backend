import os
from os import path
import pandas as pd 
import clip 
from torch.utils.data.dataset import Dataset

class DescriptionDataset(Dataset):
    """
    """

    def __init__(self, name = 'trecvid', root = None, subset = 'train'):
        self.root = root
        self.subset = subset 
        self.short_text_descriptions = path.join(root, f'{subset}_text_descriptions.csv')  
        
        desc_csv = pd.read_csv(self.short_text_descriptions) 
        
        self.description = []
        self.videos = set()
        for i, row in desc_csv.iterrows():
            vid_id = str(row['video_id']).zfill(5)
            self.videos.add(vid_id)
            try:
                curr_description = row['description'] 
                
                if len(curr_description) > 70:
                    curr_description = ' '.join(curr_description.split()[:70])
                self.description.append({
                    "description": clip.tokenize(curr_description)[0]    ,
                    "video_id": int(vid_id)
                })
            except:
                pass 
            for i in range(10):
                try:
                    curr_description = row[f'description_{i}'] 
                    if len(curr_description) > 70:
                        curr_description = ' '.join(curr_description.split()[:70])
                
                    self.description.append({
                        "description":  clip.tokenize(curr_description)[0]  ,
                        "video_id": int(vid_id)
                    })
                except:
                    pass
                 
        print('%d descriptions of %d videos accepted in %s.' % (len(self.description), len(self.videos), self.root))
        
    def __getitem__(self, idx):
        return self.description[idx]

    def __len__(self):
        return len(self.description)