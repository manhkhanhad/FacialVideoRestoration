# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp
import math
import mmcv
from mmcv.runner import auto_fp16
from torch.nn import functional as F
from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_component, build_loss
from ..registry import MODELS
from .basic_restorer import BasicRestorer
from torchvision.ops import roi_align
import torch
import cv2
import numpy as np

@MODELS.register_module()
class STERR_GAN(BasicRestorer):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 discriminator,
                 pixel_loss,
                 l1_loss,
                 pyramid_loss,
                 perceptual_loss,
                 gan_loss,
                 gan_componen_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, discriminator, pixel_loss, l1_loss, pyramid_loss, perceptual_loss, gan_loss, gan_componen_loss, train_cfg, test_cfg,
                         pretrained)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # discriminator
        self.discriminator = build_component(discriminator['net_d'])
        self.net_d_left_eye = build_component(discriminator['net_d_left_eye'])
        self.net_d_right_eye = build_component(discriminator['net_d_right_eye'])
        self.net_d_mouth = build_component(discriminator['net_d_mouth'])
        self.net_identity = build_component(discriminator['net_identity'])

        # self.gfpgan = build_backbone(gfpgan)
        # self.gfpgan = GFPGANv1(**gfpgan)
        # breakpoint()
        # for k,v in self.gfpgan.named_parameters(): 
        #     print(k)
        # exit()

        # self.gfpgan.eval()
        # for k in self.gfpgan.parameters():
        #     k.requires_grad = False

        # loss
        self.pixel_loss = build_loss(pixel_loss)
        self.l1_loss = build_loss(l1_loss)
        self.perceptual_loss = build_loss(perceptual_loss)
        self.gan_loss = build_loss(gan_loss)
        self.gan_component_loss = build_loss(gan_componen_loss)
        
        self.current_iter = 0
        self.pyramid_loss_weight = pyramid_loss.get('pyramid_loss_weight', 0)
        self.remove_pyramid_loss = pyramid_loss.get('remove_pyramid_loss', float('inf'))
        self.log_size = int(math.log(generator['gfpgan']['out_size'], 2))

        self.use_facial_disc = True
        self.face_ratio = generator['gfpgan']['out_size'] / 512
    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)
    
    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        output = self.generator(lq)
        
        return output

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # outputs = self(**data_batch, test_mode=False)
        # outputs = self(**data_batch, test_mode=False)
        lq = data_batch.get('lq')
        gt = data_batch.get('gt')
        facial_component = data_batch.get('facial_component')
        meta = data_batch.get('meta')
        #OPTIMIZE GENERATOR
        #Freeze disciminators
        for p in self.discriminator.parameters():
            p.requires_grad = False
        optimizer['generator'].zero_grad()

        # do not update facial component net_d
        if self.use_facial_disc:
            for p in self.net_d_left_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_right_eye.parameters():
                p.requires_grad = False
            for p in self.net_d_mouth.parameters():
                p.requires_grad = False
        
        # image pyramid loss weight#Opt
        if self.pyramid_loss_weight > 0 and self.current_iter > self.remove_pyramid_loss:
            self.pyramid_loss_weight = 1e-12  # very small weight to avoid unused param error
        if self.pyramid_loss_weight > 0:
            output, out_rgbs = self.generator(lq, return_rgb=True)
            pyramid_gt = self.construct_img_pyramid(gt)
        else:
            output, out_rgbs =  self.generator(lq, return_rgb=False)
            
        left_eyes, right_eyes, mouths = self.get_roi_regions(meta, gt, output, facial_component)
        
        #Calculate loss
        #Backward
        
        #OPTIMIZE DISCRIMINATOR
        #Unfreeze disciminators
        #Calculate loss
        #Update

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def construct_img_pyramid(self, gt):
        """Construct image pyramid for intermediate restoration loss"""
        
        gt = gt.flatten(0,1)
        '''
            gt shape = [b, l, c, w, h]
            [[f1_1, f1_2, f1_3, f1_4], 
             [f2_1, f2_2, f2_3, f2_4]
            ]
            after fatten gt shape = [bxl, c, w, h]
            [f1_1, f1_2, f1_3, f1_4, f2_1, f2_2, f2_3, f2_4] 
        
        check:  (gt.flatten(0,1) == torch.cat(gt[0,:,:,:,:].split(1) + gt[1,:,:,:,:].split(1), 0)).all()
        '''

        pyramid_gt = [gt]
        down_img = gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def get_roi_regions(self, meta, gt, output,  facial_components, eye_out_size=80, mouth_out_size=120):
        
        eye_out_size *= self.face_ratio
        mouth_out_size *= self.face_ratio
        eye_out_size = int(eye_out_size)
        mouth_out_size = int(mouth_out_size)
        
        #TODO:
        # get facial component for batchsize more than 2

        rois_eyes = []
        rois_mouths = []
        
        num_batch = len(facial_components[0]['left_eye'][0])
        
        components = []
        
        img = cv2.imread(meta[0]['gt_path'][0])
        print(img.shape)
        scale_h = img.shape[0] / 256
        scale_w = img.shape[1] / 256
        
        
        img =  gt[0][0].permute(1,2,0).cpu().numpy()*255
        img =  np.ascontiguousarray(img, dtype=np.uint8)
        
        for t in range(len(facial_components)):
            for component in ['left_eye', 'right_eye', 'mouth']:
                # components.append(torch.cat(facial_components[t][component]).reshape(4,2).T)
                for b in range(1):
                    print(scale_w, scale_h)
                    x = facial_components[t][component][0][b] / scale_w
                    w = facial_components[t][component][2][b] / scale_w
                    y = facial_components[t][component][1][b] / scale_h
                    h = facial_components[t][component][3][b] / scale_h
                    
                    cv2.rectangle(img, (int(x-w), int(y-h)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/img.jpg",img)
        exit()
        components = torch.tensor(torch.cat(components))
        
        rearrange_components = []
        for b in range(num_batch):
            rearrange_components.append(components[b::num_batch,])
        components = torch.cat(rearrange_components)
    
        for b in range(num_batch):
            for t in range(len(facial_components)):
                # img = cv2.imread(meta[b][t])
                img =  gt[b][t].permute(1,2,0).cpu().numpy()*255
                img =  np.ascontiguousarray(img, dtype=np.uint8)
                breakpoint()
                
                for i in range(18):
                    loc = components[i, :]
                    x,y = loc[0:2]
                    w,h = loc[2:]
                    
                    x = x / 2
                    w = w / 2
                    y = y / 2
                    h = h / 2
                    cv2.rectangle(img, (int(x-w), int(y-h)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                # cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/{}_{}".format(b,t),img)
                cv2.imwrite("/home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/test_crop_facial_componen/img.jpg",img)
        exit()
        
        for i in range(num_batch):
            for t in range(len()):
                loc_left_eyes = facial_component.get('loc_left_eyes')
                loc_right_eyes = facial_component.get('loc_right_eyes')
                loc_mouths = facial_component.get('loc_mouths')
                
                eye_out_size *= self.face_ratio
                mouth_out_size *= self.face_ratio

                for b in range(loc_left_eyes.size(0)):  # loop for batch size
                    # left eye and right eye
                    img_inds = loc_left_eyes.new_full((2, 1), b)
                    bbox = torch.stack([loc_left_eyes[b, :], loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
                    rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
                    rois_eyes.append(rois)
                    # mouse
                    img_inds = loc_left_eyes.new_full((1, 1), b)
                    rois = torch.cat([img_inds, loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
                    rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)
        
        #Code check get correct facial component
        

        # real images
        all_eyes = roi_align(gt, boxes=rois_eyes, output_size=eye_out_size) * self.face_ratio
        left_eyes_gt = all_eyes[0::2, :, :, :]
        right_eyes_gt = all_eyes[1::2, :, :, :]
        mouths_gt = roi_align(gt, boxes=rois_mouths, output_size=mouth_out_size) * self.face_ratio
        # output
        all_eyes = roi_align(output, boxes=rois_eyes, output_size=eye_out_size) * self.face_ratio
        left_eyes = all_eyes[0::2, :, :, :]
        right_eyes = all_eyes[1::2, :, :, :]
        mouths = roi_align(output, boxes=rois_mouths, output_size=mouth_out_size) * self.face_ratio

        return left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes, right_eyes, mouths