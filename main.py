import argparse
import os
from dataset import get_loader
from solver import Solver
import time



def main(config):
    if config.mode == 'train':
        train_loader,val_loader = get_loader(config)
        print('train dataset loaded',len(train_loader),len(val_loader))
        if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
            os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        train = Solver(train_loader,val_loader, None,config)
        train.train()
    elif config.mode == 'test':
        #get_test_info(config)
        test_loader,_ = get_loader(config, mode='test')
        if not os.path.exists(config.test_folder): os.makedirs(config.test_folder)
        test = Solver(None, None,test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    resnet101_path = './pretrained/resnet101-5d3b4d8f.pth'
    resnet50_path = './pretrained/resnet50-19c8e357.pth'
    vgg16_path = './pretrained/vgg16-397923af.pth'
    conformer_path='./pretrained/Conformer_small_patch16.pth'
    cswin_path='./pretrained/cswin_large_384.pth'
    densenet161_path = './pretrained/densenet161-8d451a50.pth'
    pretrained_path = {'resnet101': resnet101_path, 'resnet50': resnet50_path, 'vgg16': vgg16_path,
                       'densenet161': densenet161_path,'conformer':conformer_path,'cswin':cswin_path}

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00001)  # Learning rate resnet:4e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')

    # Training settings
    parser.add_argument('--arch', type=str, default='cswin'
                        , choices=['resnet', 'vgg','densenet','conformer','cswin'])  # resnet, vgg or densenet
    parser.add_argument('--pretrained_model', type=str, default=pretrained_path)  # pretrained backbone model
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size_val', type=int, default=1)  # only support 1 now
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--load', type=str, default='')  # pretrained JL-DCF model
    parser.add_argument('--save_folder', type=str, default='checkpoints/')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--network', type=str, default='cswin'
                        , choices=['resnet50', 'resnet101', 'vgg16', 'densenet161','conformer','cswin'])  # Network Architecture
    #conformer setting
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--channel_ratio', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    # Train data
    parser.add_argument('--train_root', type=str, default='../RGBDcollection')
    parser.add_argument('--train_list', type=str, default='../RGBDcollection/train.lst')
    parser.add_argument('--img_folder', type=str, default='../DUTS-TR/DUTS-TR-Image')
    parser.add_argument('--gt_folder', type=str, default='../DUTS-TR/DUTS-TR-Mask')
    

    # Testing settings
    parser.add_argument('--model', type=str, default='./checkpoints/demo-20-1/demo-20/epoch_20.pth')  # Snapshot
    parser.add_argument('--test_folder', type=str, default='test/LFSD/')  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='LFSD',
                        choices=['NJU2K', 'NLPR', 'STERE', 'RGBD135', 'LFSD', 'SIP', 'ReDWeb-S'])  # Test image dataset
    parser.add_argument('--test_root', type=str, default='../testsod/LFSD/LFSD')
    parser.add_argument('--test_list', type=str, default='../testsod/LFSD/LFSD/test.lst')
    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)



    main(config)
