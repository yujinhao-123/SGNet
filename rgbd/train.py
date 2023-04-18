import os
import sys
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import data

from apex import amp
from torch.utils.data import DataLoader
from SGNet_vgg16 import SGNet_vgg16
from SGNet_res50 import SGNet_res50

def bce_iou_loss(pred, mask):
    bce   = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou   = 1-(inter+1)/(union-inter+1)

    return (bce+iou).mean()

def train(Dataset, Network, Backbone):
    ## dataset

    cfg    = Dataset.Config(datapath='/home/ipal/datasets/DUTS_train', savepath='./models', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=32)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    num_params = 0
    base, head = [], []
    for name, param in net.named_parameters():
        num_params += param.numel()
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    print('SGNet_' + Backbone + ' Structure')
    print(net)
    print("The number of parameters: {}".format(num_params))

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, depth, mask) in enumerate(loader):
            image, depth, mask = image.cuda().float(), depth.cuda().float(), mask.cuda().float()
            out1, out2, out3, out4, out5 = net(image, depth)

            loss1 = bce_iou_loss(out1, mask)
            loss2 = bce_iou_loss(out2, mask)
            loss3 = bce_iou_loss(out3, mask)
            loss4 = bce_iou_loss(out4, mask)
            loss5 = bce_iou_loss(out5, mask)
            loss  = loss1 + 0.8*loss2 + 0.6*loss3 + 0.4*loss4 + 0.2*loss5

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            global_step += 1
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))
                print('loss1=%.6f'%(loss1.item()))
                print('loss2=%.6f'%(loss2.item()))
                print('loss3=%.6f'%(loss3.item()))
                print('loss4=%.6f'%(loss4.item()))
                print('loss5=%.6f'%(loss5.item()))

        if (epoch + 1) % 32 == 0:
            if not os.path.exists(cfg.savepath):
                os.makedirs(cfg.savepath)
            if Backbone == 'res50':
                torch.save(net.state_dict(), cfg.savepath+'/SGNet_res50.' + str(epoch+1) + '.pth')
            elif Backbone == 'vgg16':
                torch.save(net.state_dict(), cfg.savepath+'/SGNet_vgg16.' + str(epoch+1) + '.pth')


if __name__=='__main__':
    ## set backbone network: res50 or vgg16
    backbone = 'res50'
    if backbone == 'res50':
        train(data, SGNet_res50, backbone)
    elif backbone == 'vgg16':
        train(data, SGNet_vgg16, backbone)
