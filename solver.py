import torch
from torch.nn import functional as F
from RGBDincomplete import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse1 = (192,192)
size_coarse2 = (96,96)
size_coarse3 = (48,48)
size_coarse4 = (24,24)
from tqdm import trange, tqdm




class Solver(object):
    def __init__(self, train_loader,val_loader, test_loader, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.RGBDInModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        #self.print_network(self.net, 'Incomplete modality RGBD SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                preds = self.net(images)
                #print(preds.shape)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_rgbonly.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size_train
        
        loss_vals=  []
        self.net.train()
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in tqdm(enumerate(self.train_loader)):
                sal_image, sal_label= data_batch[0], data_batch[1]

             
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image,  sal_label= sal_image.to(device),sal_label.to(device)
              
                self.optimizer.zero_grad()
               
                sal_rgb_only,sal1,sal2,sal3,sal4 = self.net(sal_image)
                
                sal_label_coarse1 = F.interpolate(sal_label, size_coarse1, mode='bilinear', align_corners=True)
                sal_label_coarse2 = F.interpolate(sal_label, size_coarse2, mode='bilinear', align_corners=True)
                sal_label_coarse3 = F.interpolate(sal_label, size_coarse3, mode='bilinear', align_corners=True)
                sal_label_coarse4 = F.interpolate(sal_label, size_coarse4, mode='bilinear', align_corners=True)
                
                sal_loss_final =  F.binary_cross_entropy_with_logits(sal_rgb_only, sal_label, reduction='sum')
                sal_loss_coarse1 = F.binary_cross_entropy_with_logits(sal1, sal_label_coarse1, reduction='sum')
                sal_loss_coarse2 = F.binary_cross_entropy_with_logits(sal2, sal_label_coarse2, reduction='sum')
                sal_loss_coarse3 = F.binary_cross_entropy_with_logits(sal3, sal_label_coarse3, reduction='sum')
                sal_loss_coarse4 = F.binary_cross_entropy_with_logits(sal4, sal_label_coarse4, reduction='sum')

                sal_rgb_only_loss = sal_loss_final + sal_loss_coarse1 + sal_loss_coarse2 + sal_loss_coarse3 + sal_loss_coarse4
                r_sal_loss += sal_rgb_only_loss.data
                r_sal_loss_item+=sal_rgb_only_loss.item() * sal_image.size(0)
                sal_rgb_only_loss.backward()
                self.optimizer.step()

 
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            print('learning rate',self.optimizer.param_groups[0]['lr'])
            # Evaluate the model on the validation set
            self.net.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for i, data_batch in tqdm(enumerate(self.val_loader)):
                    valid_image, valid_label= data_batch[0], data_batch[1]

   
                    if self.config.cuda:
                        device = torch.device(self.config.device_id)
                        valid_image, valid_label=valid_image.to(device),valid_label.to(device)
            
                    valid_rgb_only = self.net(valid_image)
                    valid_rgb_only_loss =  F.binary_cross_entropy_with_logits(valid_rgb_only,valid_label, reduction='sum')
                                  
                    running_val_loss+=valid_rgb_only_loss.item() * valid_image.size(0)
 
            # Calculate validation loss
            val_epoch_loss = running_val_loss / len(self.val_loader.dataset)
            print(f'Validation Loss: {val_epoch_loss:.6f}')
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        
