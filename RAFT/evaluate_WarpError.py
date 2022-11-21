#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
sys.path.append('core')
from networks.resample2d_package.resample2d import Resample2d
from utils.flownet import read_img, read_flo, img2tensor, tensor2img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    
    ### testing options
    opts = parser.parse_args()
    opts.cuda = True

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = Resample2d().to(device)

    print(opts)
    ### load image list
    input_folder_root = "/home/ldtuan/VideoRestoration/fast_blind_video_consistency/data/input"
    fw_flow_folder_root = "/home/ldtuan/VideoRestoration/fast_blind_video_consistency/data/output/fw_flow"
    fw_occ_folder_root = "/home/ldtuan/VideoRestoration/fast_blind_video_consistency/data/output/fw_occlusion"
    fw_flow_rgb_folder_root = "/home/ldtuan/VideoRestoration/fast_blind_video_consistency/data/output/fw_flow_rgb"

    input_folders = [os.path.join(input_folder_root, folder) for folder in os.listdir(input_folder_root)]
    fw_flow_folders = [os.path.join(fw_flow_folder_root, folder) for folder in os.listdir(input_folder_root)]
    fw_occ_folders = [os.path.join(fw_occ_folder_root, folder) for folder in os.listdir(input_folder_root)]
    fw_flow_rgb_folders = [os.path.join(fw_flow_rgb_folder_root, folder) for folder in os.listdir(input_folder_root)]

    err_all = {}
    for input_folder, fw_flow_folder, fw_occ_folder, fw_flow_rgb_folder in zip(input_folders, fw_flow_folders, fw_occ_folders, fw_flow_rgb_folders):

        frame_list = glob.glob(os.path.join(input_folder, "*.png"))
        err = 0
        for t in range(1, len(frame_list)):
            ### load input images
            filename = os.path.join(input_folder, "%03d.png" %(t - 1)) 
            img1 = read_img(filename)
            filename = os.path.join(input_folder, "%03d.png" %(t))
            img2 = read_img(filename)

            print("Evaluate Warping Error on video %s, %s" %(input_folder.split("/")[-1], filename))


            ### load flow
            filename = os.path.join(fw_flow_folder, "%05d.flo" %(t-1))
            flow = read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(fw_occ_folder, "%05d.png" %(t-1))
            occ_mask = read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = img2tensor(img2).to(device)
                flow = img2tensor(flow).to(device)

                ## warp img2
                warp_img2 = flow_warping(img2, flow)

                ## convert to numpy array
                warp_img2 = tensor2img(warp_img2)


            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            # breakpoint()
            
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err += np.sum(np.square(diff)) / N

        err_all[input_folder] = err / (len(frame_list) - 1)
    
    print (err_all)
