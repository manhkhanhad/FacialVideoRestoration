import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import math
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.flownet import detect_occlusion, resize_flow, tensor2img, read_img, save_img
import matplotlib.pyplot as plt

#  python compute_flow_occlusion.py --model=models/raft-things.pth --path=demo-frames
#  python compute_flow_occlusion.py --model=models/raft-things.pth --path=/home/ldtuan/VideoRestoration/fast_blind_video_consistency/data/input/-cNe_z2qsGQ_0000_S1010_E1114_L893_T112_R1213_B432_e2e
#  python compute_flow_occlusion.py --model=models/raft-things.pth --path=/home/ldtuan/VideoRestoration/STERR-GAN/RAFT/data/5u-Aw6NOIy0_0000_S125_E207_L645_T62_R1029_B446_e2e

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img1, img2, flo, image_name):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('estimated optical flow')
    ax3.imshow(flo)
    plt.savefig(f"/home/ldtuan/VideoRestoration/STERR-GAN/RAFT/output2/{image_name}")
    # plt.show()

def viz2(img1, img2, flo, image_name):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    # flo = cv2.cvtColor(flo, cv2.COLOR_GRAY2RGB)
    # flo = flow_viz.flow_to_image(flo)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('occlusion mask')
    ax3.imshow(flo, cmap='gray')
    plt.savefig(f"/home/ldtuan/VideoRestoration/STERR-GAN/RAFT/output/{image_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    cnt = 0 
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # flo = flow_viz.flow_to_image(flow_up[0].permute(1,2,0).cpu().numpy())
            # print ("image: ", image1.shape, image1[0].permute(1,2,0).shape)
            # print ("flow_low: ", flow_low.shape, flow_low[0].permute(1,2,0).shape)
            # print ("flow_up: ", flow_up.shape, flow_up[0].permute(1,2,0).shape)
            # print ("flo: ", flo.shape)
            print("image1", imfile1)
            print("image2", imfile2)
            ### resize image
            print ("image1", image1.shape)
            size_multiplier = 64
            H_orig = image1.shape[2]
            W_orig = image1.shape[3]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)

            viz(image1, image2, flow_up, imfile1.split("/")[-1])

            fw_flow = tensor2img(flow_up)
            # fw_flow = np.concatenate([tensor2img(image1), flow_viz.flow_to_image(fw_flow)], axis=0)

            _, bw_flow = model(image2, image1, iters=20, test_mode=True)
            bw_flow = tensor2img(bw_flow)
            # bw_flow = np.concatenate([tensor2img(image2), flow_viz.flow_to_image(bw_flow)], axis=0)

            print ("image1", image1.shape)
            print ("image2", image2.shape)
            print ("flow_up", flow_up.shape)
            print ("fw_flow", fw_flow.shape)
            print ("bw_flow", bw_flow.shape)
            # print ("bw_flow", tensor2img(bw_flow))

            fw_flow = resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
            bw_flow = resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 

            print ("fw_flow", fw_flow)
            print ("fw_flow2", fw_flow.shape)
            print ("bw_flow2", bw_flow.shape)
            # fw_occ = detect_occlusion(bw_flow, fw_flow)
            fw_occ = detect_occlusion(bw_flow, fw_flow)
            print ("fw_occ", fw_occ.shape)
            print ("fw_occ", fw_occ)
            save_img(fw_occ, f"/home/ldtuan/VideoRestoration/STERR-GAN/RAFT/output2/{imfile1.split('/')[-1]}")
            # print ("fw_occ", fw_flow/255)
            # print ("bw_flow", bw_flow)
            # viz2(image1, image2, fw_occ, imfile1.split("/")[-1])
            viz(image1, image2, flow_up, imfile1.split("/")[-1])
            cnt += 1
            # if (cnt == 3):
            #     sys.exit(0)

