import torch
import torch.nn as nn
from cswin import CSWinTransformer
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
import os
import cv2
import numpy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
im_size=(320,320)
k_channels=[144,288,576,1152]

class FCU(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCU, self).__init__()

        
        #self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv_project = nn.Linear(inplanes, outplanes)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x_r = self.act(self.bn(self.conv_project(x)))

        return x_r

class RGBDInModule(nn.Module):
    def __init__(self, backbone):
        super(RGBDInModule, self).__init__()
        self.backbone = backbone
        for i in range(4):
            self.add_module('expand_block_' + str(i), FCU(k_channels[i], k_channels[i]))

        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=True)
        

    def forward(self, x):
        feature_stage=[]
        x,x1= self.backbone(x)
        '''for i in range(len(x1)):
            print("stage",i,x1[i].shape)'''
        a=[1,5,37,40]
        count=0
        for i in a:
            #print(i,'The backbone features are',x1[i].shape)
            x_r=eval('self.expand_block_' + str(count))(x1[i])
            #print(i,x_r.shape)
            count=count+1
            feature_stage.append(x_r)
            

        return feature_stage
        
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        avg_pool = self.avg_pool(x).view(batch_size, num_channels)
        fc1 = self.relu(self.fc1(avg_pool))
        fc2 = self.sigmoid(self.fc2(fc1))
        fc2 = fc2.view(batch_size, num_channels, 1, 1)
        return fc2 * x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return attention * x
    
class RGBD_incomplete(nn.Module):
    def __init__(self,RGBDInModule):
        super(RGBD_incomplete, self).__init__()
        
        self.RGBDInModule = RGBDInModule
        self.relu = nn.ReLU(inplace=True)
        self.conv_stage1=SpatialAttention(k_channels[0])
        self.conv_stage2=SpatialAttention(k_channels[1])
        self.conv_stage3=SpatialAttention(k_channels[2])
        self.conv_stage4=SpatialAttention(k_channels[3])
 
        self.deconv_stage1=nn.ConvTranspose2d(k_channels[0],1,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_stage2=nn.ConvTranspose2d(k_channels[1],1,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_stage3=nn.ConvTranspose2d(k_channels[2],1,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        self.deconv_stage4=nn.ConvTranspose2d(k_channels[3],1,kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        
        self.ca_stage1=ChannelAttention(1)
        self.ca_stage2=ChannelAttention(145)
        self.ca_stage3=ChannelAttention(289)
        self.ca_stage4=ChannelAttention(577)
        
        self.sal_stage1=nn.Sequential(nn.Conv2d(1, 1, 3,1,1), self.relu)
        self.sal_stage2=nn.Sequential(nn.Conv2d(k_channels[0]+1, 1, 1,1), self.relu)
        self.sal_stage3=nn.Sequential(nn.Conv2d(k_channels[1]+1, 1, 1,1), self.relu)
        self.sal_stage4=nn.Sequential(nn.Conv2d(k_channels[2]+1, 1, 1,1), self.relu)
        self.last_conv=nn.Conv2d(4,1,1,1)

        
    def forward(self, f_all):
        feat_rgb = self.RGBDInModule(f_all)
        #spatial attention
        rgb_branch1 = self.conv_stage1(feat_rgb[0])
        rgb_branch2 = self.conv_stage2(feat_rgb[1])
        rgb_branch3 = self.conv_stage3(feat_rgb[2])
        rgb_branch4 = self.conv_stage4(feat_rgb[3])
        
        #concatenation of adjacent features
        rgb_out4 = torch.cat((self.deconv_stage4(rgb_branch4),rgb_branch3),dim=1)
        rgb_out3 = torch.cat((self.deconv_stage3(rgb_branch3),rgb_branch2),dim=1)
        rgb_out2 = torch.cat((self.deconv_stage2(rgb_branch2),rgb_branch1),dim=1)
        rgb_out1 = self.deconv_stage1(rgb_branch1)
     
        #channel attention
        rgb_out4ca = self.ca_stage4(rgb_out4)
        rgb_out3ca = self.ca_stage3(rgb_out3)
        rgb_out2ca = self.ca_stage2(rgb_out2)
        rgb_out1ca = self.ca_stage1(rgb_out1)
        
        #saliency maps at multi scales
        rgb_sal_1=self.sal_stage1(rgb_out1ca)
        rgb_sal_2=self.sal_stage2(rgb_out2ca)
        rgb_sal_3=self.sal_stage3(rgb_out3ca)
        rgb_sal_4=self.sal_stage4(rgb_out4ca)
        
        '''print(rgb_branch1.shape,rgb_out1.shape,rgb_out1ca.shape,rgb_sal_1.shape)
        print(rgb_branch2.shape,rgb_out2.shape,rgb_out2ca.shape,rgb_sal_2.shape)
        print(rgb_branch3.shape,rgb_out3.shape,rgb_out3ca.shape,rgb_sal_3.shape)
        print(rgb_branch4.shape,rgb_out4.shape,rgb_out4ca.shape,rgb_sal_4.shape)'''
  
        # Resize tensors to have the same number of channels
        tensor_1 = torch.nn.functional.interpolate(rgb_sal_1, size=(384,384))
        tensor_2 = torch.nn.functional.interpolate(rgb_sal_2, size=(384,384))
        tensor_3 = torch.nn.functional.interpolate(rgb_sal_3, size=(384,384))
        tensor_4 = torch.nn.functional.interpolate(rgb_sal_4, size=(384,384))

        # Concatenate tensors along the channel dimension (dimension 1)
        concatenated_tensor = torch.cat((tensor_1, tensor_2, tensor_3, tensor_4), dim=1)

        feat_rgb_out=self.last_conv(concatenated_tensor)
        #print(feat_rgb_out.shape)
        
        return feat_rgb_out,rgb_sal_1,rgb_sal_2,rgb_sal_3,rgb_sal_4
        


def build_model(network='cswin', base_model_cfg='cswin'):
    backbone = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4.0)
      
   

    return RGBD_incomplete(RGBDInModule(backbone))
