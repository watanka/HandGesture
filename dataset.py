import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import torch 

class GestureRecognitionDataModule(pl.LightningDataModule) :

    def __init__(self, config, train_df, test_df, collate_fn = None) :
        super().__init__()
        
        self.config = config
        train_split, val_split, _, _ = train_test_split(train_df, train_df['label'], test_size = 0.2, random_state = config.seed)

        self.train_df = train_split
        self.val_df = val_split
        self.test_df = test_df
        self.collate_fn = collate_fn


    @property
    def train_dataset(self) :
        return GestureRecognitionDataset(config = self.config,
                                         video_path_list = self.train_df['path'].values, 
                                         label_list = self.train_df['label'].values
                                        )

    @property
    def val_dataset(self) :
        return GestureRecognitionDataset(config = self.config,
                                         video_path_list = self.val_df['path'].values, 
                                         label_list = self.val_df['label'].values
                                        )

    @property
    def test_dataset(self) :
        return GestureRecognitionDataset(config = self.config,
                                         video_path_list = self.test_df['path'].values, 
                                         label_list = self.test_df['label'].values
                                        )


    def train_dataloader(self) :
        return DataLoader(self.train_dataset, 
                          batch_size = self.config.batch_size,
                          shuffle = True,
                          num_workers = self.config.num_workers,
                          persistent_workers = self.config.num_workers > 0,
                          pin_memory = True,
                          collate_fn = self.collate_fn    
                        )

    def val_dataloader(self) :
        return DataLoader(self.val_dataset, 
                          batch_size = self.config.batch_size,
                          num_workers = self.config.num_workers,
                          persistent_workers = self.config.num_workers > 0,
                          pin_memory = True,
                          collate_fn = self.collate_fn    
                        )


    def test_dataloader(self) :
        return DataLoader(self.test_dataset, 
                          batch_size = self.config.batch_size,
                          num_workers = self.config.num_workers,
                          persistent_workers = self.config.num_workers > 0,
                          pin_memory = True,
                          collate_fn = self.collate_fn    
                        )

class GestureRecognitionDataset(Dataset) :

    def __init__(self, config, video_path_list, label_list) :
        self.config = config
        self.video_path_list = video_path_list
        self.label_list = label_list

    def __getitem__(self, index) :
        frames = self.get_video(self.video_path_list[index])
        
        if self.label_list is not None :
            label = self.label_list[index]
            return frames, label
        else :
            return frames

    def get_video(self, path) :
        frames = []
        cap = cv2.VideoCapture(path)
        for _ in range(self.config.fps) :
            _, img = cap.read()
            img = cv2.resize(img, (self.config.img_size, self.config.img_size))
            img = img / 255.
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3,0,1,2)

    def __len__(self) :
        return len(self.video_path_list)