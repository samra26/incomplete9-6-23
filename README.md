!git clone https://github.com/samra26/rgbd_incomplete_step1.git

!pip install torchsampler

!pip install einops

!pip install torchsummary

!pip install fvcore

!python main.py --train_root=/kaggle/input/rgbdcollection/RGBDcollection --train_list=/kaggle/input/rgbdcollection/RGBDcollection/train.lst --img_folder=/kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Image --gt_folder=/kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Mask --pretrained_model=/kaggle/input/cswintransformer/cswin_base_384.pth

import os
import shutil
shutil.rmtree('/kaggle/working/rgbd_incomplete_step1')
os.chdir('/kaggle/working')

stage 0 torch.Size([1, 96, 96, 96])
stage 1 torch.Size([1, 96, 96, 96])
stage 2 torch.Size([1, 192, 48, 48])
stage 3 torch.Size([1, 192, 48, 48])
stage 4 torch.Size([1, 192, 48, 48])
stage 5 torch.Size([1, 192, 48, 48])
stage 6 torch.Size([1, 384, 24, 24])
stage 7 torch.Size([1, 384, 24, 24])
stage 8 torch.Size([1, 384, 24, 24])
stage 9 torch.Size([1, 384, 24, 24])
stage 10 torch.Size([1, 384, 24, 24])
stage 11 torch.Size([1, 384, 24, 24])
stage 12 torch.Size([1, 384, 24, 24])
stage 13 torch.Size([1, 384, 24, 24])
stage 14 torch.Size([1, 384, 24, 24])
stage 15 torch.Size([1, 384, 24, 24])
stage 16 torch.Size([1, 384, 24, 24])
stage 17 torch.Size([1, 384, 24, 24])
stage 18 torch.Size([1, 384, 24, 24])
stage 19 torch.Size([1, 384, 24, 24])
stage 20 torch.Size([1, 384, 24, 24])
stage 21 torch.Size([1, 384, 24, 24])
stage 22 torch.Size([1, 384, 24, 24])
stage 23 torch.Size([1, 384, 24, 24])
stage 24 torch.Size([1, 384, 24, 24])
stage 25 torch.Size([1, 384, 24, 24])
stage 26 torch.Size([1, 384, 24, 24])
stage 27 torch.Size([1, 384, 24, 24])
stage 28 torch.Size([1, 384, 24, 24])
stage 29 torch.Size([1, 384, 24, 24])
stage 30 torch.Size([1, 384, 24, 24])
stage 31 torch.Size([1, 384, 24, 24])
stage 32 torch.Size([1, 384, 24, 24])
stage 33 torch.Size([1, 384, 24, 24])
stage 34 torch.Size([1, 384, 24, 24])
stage 35 torch.Size([1, 384, 24, 24])
stage 36 torch.Size([1, 384, 24, 24])
stage 37 torch.Size([1, 384, 24, 24])
stage 38 torch.Size([1, 768, 12, 12])
stage 39 torch.Size([1, 768, 12, 12])
stage 40 torch.Size([1, 768, 12, 12])
