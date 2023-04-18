import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, depth, mask=None):
        image = (image - self.mean)/self.std
        depth /= 255
        if mask is None:
            return image, depth
        else:
            mask /= 255
            return image, depth, mask

class RandomCrop(object):
    def __call__(self, image, depth, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], depth[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, depth, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], depth[:, ::-1], mask[:, ::-1]
        else:
            return image, depth, mask

class RandomRotate(object):
    def __call__(self, image, depth, mask=None):
        angle = np.random.randint(-10, 10)
        h, w, _ = image.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        depth = cv2.warpAffine(depth, M, (w, h))
        if mask is None:
            return image, depth
        else:
            mask = cv2.warpAffine(mask, M, (w, h))
            return image, depth, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, depth, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth  = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image, depth
        else:
            mask  = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, depth, mask

class ToTensor(object):
    def __call__(self, image, depth, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        depth = torch.from_numpy(depth)
        if mask is None:
            return image, depth
        else:
            mask = torch.from_numpy(mask)
            return image, depth, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.randomrotate = RandomRotate()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()
        image_path = self.cfg.datapath+'/imgs/'
        self.images = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg')]
        self.depth_path = self.cfg.datapath+'/depths/'
        if self.cfg.mode=='train':
            self.mask_path = self.cfg.datapath+'/gts/'

    def __getitem__(self, idx):
        image_name  = self.images[idx]
        name = image_name.split('/')[-1]
        image = cv2.imread(image_name)[:,:,::-1].astype(np.float32)
        depth_name = self.depth_path + name[:-4] + '.png'
        depth  = cv2.imread(depth_name, 0).astype(np.float32)
        shape = image.shape[:2]
        if self.cfg.mode=='train':
            mask_name = self.mask_path + name[:-4] + '.png'
            mask  = cv2.imread(mask_name, 0).astype(np.float32)
            image, depth, mask = self.normalize(image, depth, mask)
            image, depth, mask = self.randomcrop(image, depth, mask)
            image, depth, mask = self.randomflip(image, depth, mask)
            image, depth, mask = self.randomrotate(image, depth, mask)
            return image, depth, mask
        else:
            image, depth = self.normalize(image, depth)
            image, depth = self.resize(image, depth)
            image, depth = self.totensor(image, depth)
            return image, depth.unsqueeze(0), shape, name[:-4]

    def collate(self, batch):
#        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]#multi-scale 
        size = [352, 352, 352, 352, 352][np.random.randint(0, 5)]#single-scale 
        image, depth, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        depth = torch.from_numpy(np.stack(depth, axis=0)).unsqueeze(1)
        return image, depth, mask

    def __len__(self):
        return len(self.images)
