exp_name = 'basicvsr_reds4'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
    ),
    gfpgan=dict(
            # type='GFPGANv1',
            out_size=256,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=[1, 3, 3, 1],
            decoder_load_path="experiments/pretrained_models/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth",
            # pretrained="/home/VideoRestoration/VideoRestoration/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
            pretrained= "/home/VideoRestoration/VideoRestoration/GFPGAN/experiments/train_GFPGANv1_512/models/net_g_220000.pth",
            fix_decoder=True,
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))



# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRREDSMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/home/VideoRestoration/VideoRestoration/dataset/REDS/train/train_sharp_bicubic/X4',
            gt_folder='/home/VideoRestoration/VideoRestoration/dataset/REDS/train/train_sharp',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/home/VideoRestoration/VideoRestoration/dataset/REDS/val/val_sharp_bicubic',
        gt_folder='/home/VideoRestoration/VideoRestoration/dataset/REDS/val/val_sharp',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='data/test/LQ',
        gt_folder='data/test/GT',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True