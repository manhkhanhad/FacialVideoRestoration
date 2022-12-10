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
from basicsr.losses.gan_loss import r1_penalty

import math
import sys
sys.path.append("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/RAFT")
sys.path.append('/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.flownet import detect_occlusion_tensor, resize_flow, tensor2img as ts2im, read_img, save_img, img2tensor
from networks.resample2d_package.resample2d import Resample2d

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
                 discriminator = None,
                 pixel_loss = None,
                 l1_loss = None,
                 pyramid_loss = None,
                 perceptual_loss = None,
                 gan_loss = None,
                 gan_component_loss = None,
                 identity_loss = None,
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
        

        self.l1_loss = build_loss(l1_loss) if l1_loss is not None else None
        self.perceptual_loss = build_loss(perceptual_loss) if perceptual_loss is not None else None
        self.gan_loss = build_loss(gan_loss) if gan_loss is not None else None
        self.gan_component_loss = build_loss(gan_component_loss) if gan_component_loss is not None else None
        
        # self.current_iter = 1
        self.register_buffer('current_iter', torch.ones(1))
        
        if identity_loss is not None:
            self.identity_weight = identity_loss.get('identity_weight', 0)
        else:
            self.identity_weight = 0
            
        if pyramid_loss is not None:
            self.pyramid_loss_weight = pyramid_loss.get('pyramid_loss_weight', 0)
            self.remove_pyramid_loss = pyramid_loss.get('remove_pyramid_loss', float('inf'))
        else:
            self.pyramid_loss_weight, self.remove_pyramid_loss = 0, float('inf')
        
        self.log_size = int(math.log(generator['gfpgan']['out_size'], 2))
        self.use_facial_disc = True
        self.face_ratio = generator['gfpgan']['out_size'] / 512
        
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.gfp_fix_iter = train_cfg.get('gfp_fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False
        self.is_gfp_weight_fixed = False
        
        #NEED TO BE ADDED TO CONFIGS;
        self.net_d_iters = 1 # Update discriminator after net_d_iters
        self.net_d_init_iters = 0 #Begin count discriminator
        self.net_d_reg_every = 16
        self.comp_style_weight = 200
        self.r1_reg_weight = 10
        
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

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border
        convert_to = self.test_cfg.get('convert_to', None)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2img(output[:, i, :, :, :])
                    gt_i = tensor2img(gt[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](
                        output_i, gt_i, crop_border, convert_to=convert_to))
                eval_result[metric] = np.mean(avg)
            elif output.ndim == 4:  # an image: (n, c, t, w), for Vimeo-90K-T
                output_img = tensor2img(output)
                gt_img = tensor2img(gt)
                value = self.allowed_metrics[metric](
                    output_img, gt_img, crop_border, convert_to=convert_to)
                eval_result[metric] = value

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
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        with torch.no_grad():
            output, _ = self.generator(lq, return_rgb=False)
        output = output.reshape(lq.size())

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

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
            if output.ndim == 4:  # an image, key = 000001/0000 (Vimeo-90K)
                img_name = meta[0]['key'].replace('/', '_')
                if isinstance(iteration, numbers.Number):
                    save_path = osp.join(
                        save_path, f'{img_name}-{iteration + 1:06d}.png')
                elif iteration is None:
                    save_path = osp.join(save_path, f'{img_name}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), save_path)
            elif output.ndim == 5:  # a sequence, key = 000
                folder_name = meta[0]['key'].split('/')[0]
                for i in range(0, output.size(1)):
                    if isinstance(iteration, numbers.Number):
                        save_path_i = osp.join(
                            save_path, folder_name,
                            f'{i:08d}-{iteration + 1:06d}.png')
                    elif iteration is None:
                        save_path_i = osp.join(save_path, folder_name,
                                               f'{i:08d}.png')
                    else:
                        raise ValueError('iteration should be number or None, '
                                         f'but got {type(iteration)}')
                    mmcv.imwrite(
                        tensor2img(output[:, i, :, :, :]), save_path_i)

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
        
        if self.current_iter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.current_iter == self.fix_iter:
            # train all the parameters
            for k, v in self.generator.named_parameters():
                if 'spynet' in k or 'edvr' in k:
                    v.requires_grad_(True)

        #Fix GFPGAM at the beginning
        if self.current_iter < self.gfp_fix_iter:
            if not self.is_gfp_weight_fixed:
                self.is_gfp_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'gfp' in k:
                        v.requires_grad_(False)
        elif self.current_iter >= self.fix_iter:
            # train all the parameters
            # self.generator.requires_grad_(True)
            for k, v in self.generator.named_parameters():
                if 'gfp' in k:
                    v.requires_grad_(True)
                    
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
        
        # losses = dict()
        # output, _ = self.generator(lq, return_rgb = False)
        # loss_pix = self.pixel_loss(output, gt)
        # losses['loss_pix'] = loss_pix
        # outputs = dict(
        #     losses=losses,
        #     num_samples=len(gt.data),
        #     results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        
        # loss, log_vars = self.parse_losses(outputs.pop('losses'))
        # print(log_vars)

        # # optimize
        # optimizer['generator'].zero_grad()
        # loss.backward()
        # optimizer['generator'].step()

        # outputs.update({'log_vars': log_vars})
        # return outputs
        
        
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
            loss_dict['l_g_pix'] = l_g_pix.detach().cpu()

            err = torch.tensor(0.0, device=output.device)
            train_mode = True        
            images1 = output.squeeze(0)[:-1]
            images2 = output.squeeze(0)[1:]

            if (torch.all(images1 < 255) and torch.all(images1 > -255) and torch.all(images2 < 255) and torch.all(images2 > -255)):
                fw_flows, bw_flows = compute_flow_tensor(images1, images2, self.raft, train_mode)
                fw_occs = detect_occlusion_tensor(bw_flows, fw_flows, train_mode)
                fw_occs = torch.stack([fw_occs, fw_occs, fw_occs], axis=1)
                err += evaluate_warp_error_tensor(images1, images2, fw_flows, fw_occs, self.flow_warping, train_mode)
                # compute stable loss
                l_g_total += err * 1000000
                loss_dict['loss_stable'] = err.detach().cpu() * 1000000
            
            # Image pyramid loss
            if self.pyramid_loss_weight > 0:
                for i in range(0, self.log_size - 2):
                    l_pyramid = self.l1_loss(out_rgbs[i], pyramid_gt[i]) * self.pyramid_loss_weight
                    l_g_total += l_pyramid
                    loss_dict[f'l_p_{2**(i+3)}'] = l_pyramid.detach().cpu()
            
            # Perceptual loss
            if self.perceptual_loss is not None:
                l_g_percep, l_g_style = self.perceptual_loss(output, gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep.detach().cpu()
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style.detach().cpu()
            
            # Gan loss
            if self.gan_loss is not None:
                fake_g_pred = self.discriminator(output)
                l_g_gan = self.gan_loss(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan.detach().cpu()
                
            # Facial component loss
            if self.gan_component_loss is not None:
                # left eye
                fake_left_eye, fake_left_eye_feats = self.net_d_left_eye(left_eyes_output, return_feats=True)
                l_g_gan = self.gan_component_loss(fake_left_eye, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_left_eye'] = l_g_gan.detach().cpu()
                # right eye
                fake_right_eye, fake_right_eye_feats = self.net_d_right_eye(right_eyes_output, return_feats=True)
                l_g_gan = self.gan_component_loss(fake_right_eye, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_right_eye'] = l_g_gan.detach().cpu()
                # mouth
                fake_mouth, fake_mouth_feats = self.net_d_mouth(mouths_output, return_feats=True)
                l_g_gan = self.gan_component_loss(fake_mouth, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan_mouth'] = l_g_gan.detach().cpu()

                if self.comp_style_weight > 0:
                    # get gt feat
                    _, real_left_eye_feats = self.net_d_left_eye(left_eyes_gt, return_feats=True)
                    _, real_right_eye_feats = self.net_d_right_eye(right_eyes_gt, return_feats=True)
                    _, real_mouth_feats = self.net_d_mouth(mouths_gt, return_feats=True)

                    def _comp_style(feat, feat_gt, criterion):
                        return criterion(self._gram_mat(feat[0]), self._gram_mat(
                            feat_gt[0].detach())) * 0.5 + criterion(
                                self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

                    # facial component style loss
                    comp_style_loss = 0
                    comp_style_loss += _comp_style(fake_left_eye_feats, real_left_eye_feats, self.l1_loss)
                    comp_style_loss += _comp_style(fake_right_eye_feats, real_right_eye_feats, self.l1_loss)
                    comp_style_loss += _comp_style(fake_mouth_feats, real_mouth_feats, self.l1_loss)
                    comp_style_loss = comp_style_loss * self.comp_style_weight
                    l_g_total += comp_style_loss
                    loss_dict['l_g_comp_style_loss'] = comp_style_loss.detach().cpu()
            
            # Identity loss
            if self.identity_weight != 0:
                # get gray images and resize
                out_gray = self.gray_resize_for_identity(output)
                gt_gray = self.gray_resize_for_identity(gt)

                identity_gt = self.net_identity(gt_gray).detach()
                identity_out = self.net_identity(out_gray)
                l_identity = self.l1_loss(identity_out, identity_gt) * self.identity_weight
                l_g_total += l_identity
                loss_dict['l_identity'] = l_identity.detach().cpu()
            
        #Backward
        optimizer['generator'].zero_grad()
        l_g_total.backward()
        optimizer['generator'].step()

        # #OPTIMIZE DISCRIMINATOR
        # #Unfreeze disciminators
        for p in self.discriminator.parameters():
            p.requires_grad = True
        
        if self.use_facial_disc:
            for p in self.net_d_left_eye.parameters():
                p.requires_grad = True
            for p in self.net_d_right_eye.parameters():
                p.requires_grad = True
            for p in self.net_d_mouth.parameters():
                p.requires_grad = True
                
        #Calculate loss
        
        if self.gan_loss is not None:
            fake_d_pred = self.discriminator(output.detach())
            real_d_pred = self.discriminator(gt)
            l_d = self.gan_loss(real_d_pred, True, is_disc=True) + self.gan_loss(fake_d_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d.detach().cpu()
            # In WGAN, real_score should be positive and fake_score should be negative
            loss_dict['real_score'] = real_d_pred.detach().mean().cpu()
            loss_dict['fake_score'] = fake_d_pred.detach().mean().cpu()
        
            #Update discriminator
            optimizer['discriminator'].zero_grad()
            l_d.backward()
            if self.current_iter % self.net_d_reg_every == 0:
                gt.requires_grad = True
                real_pred = self.discriminator(gt)
                l_d_r1 = r1_penalty(real_pred, gt)
                l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                loss_dict['l_d_r1'] = l_d_r1.detach().mean().cpu()
                l_d_r1.backward()
            optimizer['discriminator'].step()
        
        if self.gan_component_loss is not None:
            #Update facial component discriminator
            optimizer['net_d_left_eye'].zero_grad()
            optimizer['net_d_right_eye'].zero_grad()
            optimizer['net_d_mouth'].zero_grad()
            
            fake_d_pred, _ = self.net_d_left_eye(left_eyes_output.detach())
            real_d_pred, _ = self.net_d_left_eye(left_eyes_gt)
            l_d_left_eye = self.gan_component_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_left_eye'] = l_d_left_eye.detach().cpu()
            l_d_left_eye.backward()
            # right eye
            fake_d_pred, _ = self.net_d_right_eye(right_eyes_output.detach())
            real_d_pred, _ = self.net_d_right_eye(right_eyes_gt)
            l_d_right_eye = self.gan_component_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_right_eye'] = l_d_right_eye.detach().cpu()
            l_d_right_eye.backward()
            # mouth
            fake_d_pred, _ = self.net_d_mouth(mouths_output.detach())
            real_d_pred, _ = self.net_d_mouth(mouths_gt)
            l_d_mouth = self.gan_component_loss(
                real_d_pred, True, is_disc=True) + self.gan_loss(
                    fake_d_pred, False, is_disc=True)
            loss_dict['l_d_mouth'] = l_d_mouth.detach().cpu()
            l_d_mouth.backward()

            optimizer['net_d_left_eye'].step()
            optimizer['net_d_right_eye'].step()
            optimizer['net_d_mouth'].step()
        
        # self.log_dict = self.reduce_loss_dict(loss_dict)
        gt = gt.reshape(lq.shape)
        output = output.reshape(lq.shape)
        
        self.current_iter += 1
        
        return dict(
            log_vars = loss_dict,
            num_samples = 1,
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu())
        )

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
    
    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.train_cfg['dist_train']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.train_cfg['rank'] == 0:
                    losses /= self.train_cfg['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram