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
from collections import OrderedDict

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
                 gan_component_loss,
                 identity_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, discriminator, pixel_loss, l1_loss, pyramid_loss, perceptual_loss, gan_loss, gan_component_loss, identity_loss, train_cfg, test_cfg,
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
        self.gan_component_loss = build_loss(gan_component_loss)
        
        self.current_iter = 1
        self.pyramid_loss_weight = pyramid_loss.get('pyramid_loss_weight', 0)
        self.identity_weight = identity_loss.get('identity_weight', 0)
        self.remove_pyramid_loss = pyramid_loss.get('remove_pyramid_loss', float('inf'))
        self.log_size = int(math.log(generator['gfpgan']['out_size'], 2))

        self.use_facial_disc = True
        self.face_ratio = generator['gfpgan']['out_size'] / 512
        
        #NEED TO BE ADDED TO CONFIGS;
        self.net_d_iters = 1 # Update discriminator after net_d_iters
        self.net_d_init_iters = 0 #Begin count discriminator
        
        
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
            
        left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes_output, right_eyes_output, mouths_output = self.get_roi_regions(meta, gt, output, facial_component)
        
        #Calculate loss
        l_g_total = 0
        loss_dict = OrderedDict()
        if (self.current_iter % self.net_d_iters == 0 and self.current_iter > self.net_d_init_iters):
            # Pixel loss
            l_g_pix = self.pixel_loss(output, gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
            
            # Image pyramid loss
            if self.pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.l1_loss(out_rgbs[i], pyramid_gt[i]) * self.pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid
            
            # Perceptual loss
            l_g_percep, l_g_style = self.perceptual_loss(output, gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style
            
            # Gan loss
            fake_g_pred = self.discriminator(output)
            l_g_gan = self.gan_loss(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
            #TODO: add facial component loss
            
            # Identity loss
            
            # get gray images and resize
            out_gray = self.gray_resize_for_identity(output)
            gt_gray = self.gray_resize_for_identity(gt)

            identity_gt = self.net_identity(gt_gray).detach()
            identity_out = self.net_identity(out_gray)
            l_identity = self.l1_loss(identity_out, identity_gt) * self.identity_weight
            l_g_total += l_identity
            loss_dict['l_identity'] = l_identity
        
        #Backward
        optimizer['generator'].zero_grad()
        l_g_total.backward()
        optimizer['generator'].step()

        
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
        pyramid_gt = [gt]
        down_img = gt
        for _ in range(0, self.log_size - 3):
            down_img = F.interpolate(down_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            pyramid_gt.insert(0, down_img)
        return pyramid_gt

    def get_roi_regions(self, meta, gt, output,  facial_components, eye_out_size=80, mouth_out_size=120):
        
        #Code check get_roi_regions
        # tests/test_models/test_facial_component.py
        
        eye_out_size *= self.face_ratio
        mouth_out_size *= self.face_ratio
        eye_out_size = int(eye_out_size)
        mouth_out_size = int(mouth_out_size)

        num_batch = facial_components.shape[0]
        facial_components = facial_components.flatten(0,1)
        
        index = torch.ones(gt.shape[0], 3).to(facial_components)
        index = (index.cumsum(0)- index).reshape(-1, 1)
        facial_components = torch.cat([index, facial_components], 1)

        left_eyes_coordinate = facial_components[0::3,:].float()
        left_eyes_gt = roi_align(gt, boxes=left_eyes_coordinate, output_size=eye_out_size)
        left_eyes_output = roi_align(output, boxes=left_eyes_coordinate, output_size=eye_out_size)
        
        right_eyes_coordinate = facial_components[1::3,:].float()
        right_eyes_gt = roi_align(gt, boxes=right_eyes_coordinate, output_size=eye_out_size)
        right_eyes_output = roi_align(output, boxes=right_eyes_coordinate, output_size=eye_out_size)
        
        mouth_coordinate = facial_components[2::3,:].float()
        mouths_gt = roi_align(gt, boxes=mouth_coordinate, output_size=mouth_out_size)
        mouths_output = roi_align(output, boxes=mouth_coordinate, output_size=mouth_out_size)

        return left_eyes_gt, right_eyes_gt, mouths_gt, left_eyes_output, right_eyes_output, mouths_output

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray