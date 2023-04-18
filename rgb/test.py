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
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./models/SGNet_' + Backbone + 'resnet50.32.pth', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0

            for image, shape, name in self.loader:
                image = image.cuda().float()
                torch.cuda.synchronize()
                time_start = time.time()
                out = self.net(image)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                res = F.interpolate(out[0], shape, mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = 255 * res
#                res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
                save_path  = '/home/ipal/evaluation/SaliencyMaps/'+ self.cfg.datapath.split('/')[-1]+'/rgb_VGG_SIP/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+name[0]+'.png', res)
            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))

if __name__=='__main__':
    for path in ['/home/ipal/datasets/SIP']:#'/home/ipal/datasets/PASCALS', '/home/ipal/datasets/ECSSD', '/home/ipal/datasets/HKU-IS', '/home/ipal/datasets/DUTS', '/home/ipal/datasets/DUT-OMRON'
        ## set backbone network: res50 or vgg16
        backbone = 'vgg16'
        if backbone == 'res50':
            test = Test(data, SGNet_res50, path, backbone)
        elif backbone == 'vgg16':
            test = Test(data, SGNet_vgg16, path, backbone)
        test.save()
