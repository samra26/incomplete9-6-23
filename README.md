!git clone https://github.com/samra26/rgbd_incomplete_step1.git

!pip install torchsampler

!python main.py --train_root=/kaggle/input/rgbdcollection/RGBDcollection --train_list=/kaggle/input/rgbdcollection/RGBDcollection/train.lst --img_folder=/kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Image --gt_folder=/kaggle/input/duts-saliency-detection-dataset/DUTS-TR/DUTS-TR-Mask --pretrained_model=/kaggle/input/cswintransformer/cswin_base_384.pth
