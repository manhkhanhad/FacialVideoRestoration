# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer

import math
import sys
sys.path.append("/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/RAFT")
sys.path.append('/mmlabworkspace/WorkSpaces/danhnt/tuyensh/khanhngo/VideoRestoration/VideoRestoration/STERR-GAN/RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.flownet import detect_occlusion_tensor, resize_flow, tensor2img, read_img, save_img, img2tensor
from networks.resample2d_package.resample2d import Resample2d

def compute_flow_tensor(image1, image2, model=None, train_mode=False):
    with torch.set_grad_enabled(train_mode):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        _, fw_flow_up = model(image1, image2, iters=20, test_mode=True)
        _, bw_flow_up = model(image2, image1, iters=20, test_mode=True)

    return fw_flow_up, bw_flow_up

def evaluate_warp_error_tensor(image1, image2, flow, occ_mask, flow_warping, train_mode=False):
    noc_mask = 1 - occ_mask
    with torch.set_grad_enabled(train_mode):
        warp_image2 = flow_warping(image2, flow)

    ## compute warping error
    diff = torch.multiply(warp_image2 - image1, noc_mask)
    N = torch.sum(noc_mask)
    print (f"N: {N}")
    if (N >= 1000):
        breakpoint()
    if N == 0:
        N = diff.shape[0] * diff.shape[1] * diff.shape[2] * diff.shape[3]
    print (f"torch.sum(torch.square(diff)) / N: {torch.sum(torch.square(diff)) / N}")
    return torch.sum(torch.square(diff)) / N

@MODELS.register_module()
class BasicVSR(BasicRestorer):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        ensemble (dict): Config for ensemble. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                #  gfpgan,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.gfp_fix_iter = train_cfg.get('gfp_fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False
        self.is_gfp_weight_fixed = False
        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmedit.models.common.ensemble import \
                    SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            for k, v in self.generator.named_parameters():
                if 'spynet' in k or 'edvr' in k:
                    v.requires_grad_(True)

        #Fix GFPGAM at the beginning
        if self.step_counter < self.gfp_fix_iter:
            if not self.is_gfp_weight_fixed:
                self.is_gfp_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'gfp' in k:
                        v.requires_grad_(False)
        elif self.step_counter >= self.fix_iter:
            # train all the parameters
            # self.generator.requires_grad_(True)
            for k, v in self.generator.named_parameters():
                if 'gfp' in k:
                    v.requires_grad_(True)


        lq = data_batch.get('lq')
        gt = data_batch.get('gt')
        H_orig, W_orig = lq.shape[-2], lq.shape[-1]

        model = self.generator
        losses = dict()
        output, flows_forward, flows_backward = self.generator(lq)
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
            losses['loss_stable'] = err * 20000
            loss_pix = self.pixel_loss(output, gt)
            losses['loss_pix'] = loss_pix
        else:
            outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
            return outputs


        # for image1, image2 in zip(output[0,:-1,:,:,:], output[0,1:,:,:,:]):
        #     if (torch.any(image1 > 1) or torch.any(image2 > 1)):
        #         continue
        #     image1 = image1.unsqueeze(0)
        #     image2 = image2.unsqueeze(0)
        #     padder = InputPadder(image1.shape)
        #     image1, image2 = padder.pad(image1, image2)
        #     # compute optical flow
        #     fw_flow, bw_flow = compute_flow_tensor(image1, image2, self.raft, train_mode)
        #     # Compute occlusion mask
        #     fw_occ = detect_occlusion_tensor(bw_flow, fw_flow, train_mode)
        #     fw_occ = torch.stack([fw_occ, fw_occ, fw_occ], axis=1)
        #     # compute warping error
        #     err += evaluate_warp_error_tensor(image1, image2, fw_flow, fw_occ, self.flow_warping, train_mode)
        
        # loss_pix = self.pixel_loss(output, gt)
        # losses['loss_pix'] = loss_pix

        print ("================")
        print (lq.shape)
        print (losses)

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

        # outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        # optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

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
            if self.forward_ensemble is not None:
                output = self.forward_ensemble(lq, self.generator)
            else:
                output = self.generator(lq)

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
