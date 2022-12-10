import sys
sys.path.append('core')

import argparse, os, cv2, glob, numpy as np, torch, math, matplotlib.pyplot as plt, torch.nn as nn
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.flownet import detect_occlusion, detect_occlusion_tensor, resize_flow, tensor2img, read_img, save_img, img2tensor
from networks.resample2d_package.resample2d import Resample2d
# python evaluate_video.py --model=models/raft-things.pth 

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def compute_flow(image1, image2, model, train_mode=False):
    H_orig, W_orig = image1.shape[2], image1.shape[3]

    with torch.set_grad_enabled(train_mode):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, fw_flow_up = model(image1, image2, iters=20, test_mode=True)
        fw_flow = tensor2img(fw_flow_up)
        _, bw_flow_up = model(image2, image1, iters=20, test_mode=True)
        bw_flow = tensor2img(bw_flow_up)

        fw_flow = resize_flow(fw_flow, W_out = W_orig, H_out = H_orig)
        bw_flow = resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 

        # fw_occ = detect_occlusion(bw_flow, fw_flow)
    return fw_flow, bw_flow

def evaluate_warp_error(image1, image2, flow, occ_mask, flow_warping, train_mode=False):
    noc_mask = 1 - occ_mask
    # breakpoint()
    with torch.set_grad_enabled(train_mode):
        flow = img2tensor(flow).to(DEVICE)
        warp_image2 = flow_warping(image2/255.0, flow)
        warp_image2 = tensor2img(warp_image2)

    ## compute warping error
    diff = np.multiply(warp_image2 - tensor2img(image1/255), noc_mask)
    N = np.sum(noc_mask)
    if N == 0:
        N = diff.shape[0] * diff.shape[1] * diff.shape[2]
    
    return np.sum(np.square(diff)) / N

def evaluate_video(folder_path, train_mode=False):
    images = glob.glob(os.path.join(folder_path, "*.png"))
    images = sorted(images)
    err = 0

    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        fw_flow, bw_flow = compute_flow(image1, image2, model, train_mode)
        fw_occ = detect_occlusion(bw_flow, fw_flow)
        fw_occ = np.stack([fw_occ, fw_occ, fw_occ], axis=2)
        err += evaluate_warp_error(image1, image2, fw_flow, fw_occ, flow_warping, train_mode)

    print (folder_path, err/(len(images) - 1))

def compute_flow_tensor(image1, image2, model=None, train_mode=False):
    with torch.set_grad_enabled(train_mode):
        # padder = InputPadder(image1.shape)
        # image1, image2 = padder.pad(image1, image2)
        _, fw_flow_up = model(image1, image2, iters=20, test_mode=True)
        _, bw_flow_up = model(image2, image1, iters=20, test_mode=True)

    return fw_flow_up, bw_flow_up

def evaluate_warp_error_tensor(image1, image2, flow, occ_mask, flow_warping, train_mode=False):
    noc_mask = 1 - occ_mask
    with torch.set_grad_enabled(train_mode):
        warp_image2 = flow_warping(image2/255.0, flow)

    ## compute warping error
    diff = torch.multiply(warp_image2 - image1/255.0, noc_mask)
    N = torch.sum(noc_mask)
    if N == 0:
        N = diff.shape[0] * diff.shape[1] * diff.shape[2] * diff.shape[3]
    
    return torch.sum(torch.square(diff)) / N

def evaluate_video_tensor(folder_path, train_mode=False):
    images = glob.glob(os.path.join(folder_path, "*.png"))
<<<<<<< HEAD
    images = sorted(images)
=======
    images = sorted(images)[:30]
>>>>>>> 6ee45b691b0a0a284bd86c59ec0290c0c54ed4e6
    err = 0
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        fw_flow, bw_flow = compute_flow_tensor(image1, image2, model, train_mode)
        fw_occ = detect_occlusion_tensor(bw_flow, fw_flow, train_mode)
        fw_occ = torch.stack([fw_occ, fw_occ, fw_occ], axis=1)
        err += evaluate_warp_error_tensor(image1, image2, fw_flow, fw_occ, flow_warping, train_mode)
        # breakpoint()

    print (folder_path, err/(len(images) - 1))
    
def evaluate_video_tensor2(folder_path, train_mode=False):
    images = glob.glob(os.path.join(folder_path, "*.png"))
<<<<<<< HEAD
    images = sorted(images)
=======
    images = sorted(images)[:30]
>>>>>>> 6ee45b691b0a0a284bd86c59ec0290c0c54ed4e6
    err = 0
    images = torch.stack([load_image(imfile).squeeze(0) for imfile in images])
    images1 = images[:-1]
    images2 = images[1:]

    fw_flows, bw_flows = compute_flow_tensor(images1, images2, model, train_mode)
    fw_occs = detect_occlusion_tensor(bw_flows, fw_flows, train_mode)
    fw_occs = torch.stack([fw_occs, fw_occs, fw_occs], axis=1)
    err += evaluate_warp_error_tensor(images1, images2, fw_flows, fw_occs, flow_warping, train_mode)

    print (folder_path, err)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))

model = model.module
model.to(DEVICE)
model.eval()

flow_warping = Resample2d().to(DEVICE)

# input_folder_root = "./data/"
<<<<<<< HEAD
input_folder_root = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/RAFT/data/home/ldtuan/VideoRestoration/STERR-GAN/RAFT/data/"
=======
# input_folder_root = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/test/output/"
input_folder_root = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/Test/results/images/"
>>>>>>> 6ee45b691b0a0a284bd86c59ec0290c0c54ed4e6
input_folders = [os.path.join(input_folder_root, folder) for folder in os.listdir(input_folder_root)]
input_folders.sort()

import time
for input_folder in input_folders[:6]:
    x = time.time()
<<<<<<< HEAD
    evaluate_video_tensor2(input_folder, train_mode=False)
    print (f"{time.time() - x}s")
    x = time.time()
    evaluate_video_tensor(input_folder)
=======
    # evaluate_video_tensor2(input_folder, train_mode=False)
    # print (f"{time.time() - x}s")
    # x = time.time()
    evaluate_video(input_folder)
>>>>>>> 6ee45b691b0a0a284bd86c59ec0290c0c54ed4e6
    print (f"{time.time() - x}s")



<<<<<<< HEAD
imfile1 = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/images/-acp9Dk7Ve4_0000_S223_E320_L799_T97_R1151_B449/041.png"
imfile2 = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/images/-acp9Dk7Ve4_0000_S223_E320_L799_T97_R1151_B449/042.png"

image1 = load_image(imfile1)
image2 = load_image(imfile2)

train_mode = False
fw_flow, bw_flow = compute_flow_tensor(image1, image2, model)
fw_occ = detect_occlusion_tensor(bw_flow, fw_flow, train_mode)
fw_occ = torch.stack([fw_occ, fw_occ, fw_occ], axis=1)
evaluate_warp_error_tensor(image1, image2, fw_flow, fw_occ, flow_warping)
=======
# imfile1 = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/images/-acp9Dk7Ve4_0000_S223_E320_L799_T97_R1151_B449/041.png"
# imfile2 = "/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/data/images/-acp9Dk7Ve4_0000_S223_E320_L799_T97_R1151_B449/042.png"

# image1 = load_image(imfile1)
# image2 = load_image(imfile2)

# train_mode = False
# fw_flow, bw_flow = compute_flow_tensor(image1, image2, model)
# fw_occ = detect_occlusion_tensor(bw_flow, fw_flow, train_mode)
# fw_occ = torch.stack([fw_occ, fw_occ, fw_occ], axis=1)
# evaluate_warp_error_tensor(image1, image2, fw_flow, fw_occ, flow_warping)
>>>>>>> 6ee45b691b0a0a284bd86c59ec0290c0c54ed4e6



# img1 = read_img(imfile1)
# img2 = read_img(imfile2)

# print (image1.shape)
# print (img2tensor(img1).to(DEVICE).shape)

# print (image1/255)
# print (img2tensor(img1).to(DEVICE))
# print ((image1/255 - img2tensor(img1).to(DEVICE) <= 0.00000005))

