import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data

from torch.utils.data import DataLoader
from SGNet_vgg16 import SGNet_vgg16
from SGNet_res50 import SGNet_res50

class Test(object):
    def __init__(self, Dataset, Network, Path, Backbone):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./models/SGNet_' + Backbone + '.32.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0

            for image, depth, shape, name in self.loader:
                image, depth = image.cuda().float(), depth.cuda().float()
                torch.cuda.synchronize()
                time_start = time.time()
                res, _, _, _, _ = self.net(image, depth)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 255 * res
                save_path  = '/home/ipal/evaluation/SaliencyMaps/'+ self.cfg.datapath.split('/')[-1]+'/SGNet/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+name[0]+'.png', res)
            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))

#'/home/ipal/datasets/DUT-RGBD', '/home/ipal/datasets/LFSD', '/home/ipal/datasets/NJUD', '/home/ipal/datasets/NLPR', '/home/ipal/datasets/ReDWeb-S', '/home/ipal/datasets/RGBD135', '/home/ipal/datasets/SIP', '/home/ipal/datasets/SSD', '/home/ipal/datasets/STERE'
if __name__=='__main__':
#    for path in ['/home/ipal/datasets/DUTLF-D_test']:
    for path in ['/home/ipal/datasets/LFSD', '/home/ipal/datasets/NJUD']:
        ## set backbone network: res50 or vgg16
        backbone = 'res50'
        if backbone == 'res50':
            test = Test(data, SGNet_res50, path, backbone)
        elif backbone == 'vgg16':
            test = Test(data, SGNet_vgg16, path, backbone)
        test.save()
