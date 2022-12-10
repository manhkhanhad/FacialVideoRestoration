exp_name = 'BasicVSR_GFPGAN_facial_sr'

# model settings
model = dict(
    type='STERR_GAN',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=52, # The origin is 64, change to 32 to match the input channel in GFPGAN
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        gfpgan=dict(
                # type='GFPGANv1',
                out_size=512,
                num_style_feat=512,
                channel_multiplier=1,
                resample_kernel=[1, 3, 3, 1],
                # decoder_load_path="experiments/pretrained_models/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth",
                # pretrained= "/home/ldtuan/VideoRestoration/GFPGAN/experiments/finetune2_GFPGAN_TalkingHead_w_facial_components/models/net_g_435000.pth",
                pretrained= "/home/ldtuan/VideoRestoration/STERR-GAN/pretraineds/GFPGANv1.pth",
                fix_decoder=True,
                num_mlp=8,
                lr_mlp=0.01,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
    ),
    discriminator=dict(
        net_d=dict(
            type='StyleGAN2Discriminator',
            out_size=512,
            channel_multiplier=1,
            resample_kernel=[1, 3, 3, 1],
            pretrained="/home/ldtuan/VideoRestoration/STERR-GAN/pretraineds/GFPGANv1_net_d.pth"
        ),
        net_d_left_eye=dict(
            type='FacialComponentDiscriminator',
            pretrained="/home/ldtuan/VideoRestoration/STERR-GAN/pretraineds/GFPGANv1_net_d_left_eye.pth",
        ),
        net_d_right_eye=dict(
            type='FacialComponentDiscriminator',
            pretrained="/home/ldtuan/VideoRestoration/STERR-GAN/pretraineds/GFPGANv1_net_d_right_eye.pth",
        ),
        net_d_mouth=dict(
            type='FacialComponentDiscriminator',
            pretrained="/home/ldtuan/VideoRestoration/STERR-GAN/pretraineds/GFPGANv1_net_d_mouth.pth",
        ),
        net_identity=dict(
            type="ResNetArcFace",
            block="IRBlock",
            layers=[2, 2, 2, 2],
            use_se=False,
            pretrained="/home/ldtuan/VideoRestoration/GFPGAN/experiments/pretrained_models/arcface_resnet18.pth"
        )
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=0.1, reduction='mean'),
    l1_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    pyramid_loss=dict(pyramid_loss_weight=1,remove_pyramid_loss=500),
    perceptual_loss=dict(type="PerceptualLoss",layer_weights=dict(
                            conv1_2=0.1,
                            conv2_2=0.1,
                            conv3_4=1,
                            conv4_4=1,
                            conv5_4=1,
                         ), vgg_type="vgg19", use_input_norm=True, perceptual_weight=1.0,
                         style_weight=50, range_norm=True, criterion="l1"),
    gan_loss=dict(type="GANLoss", gan_type="wgan_softplus", loss_weight=0.1),
    gan_component_loss=dict(type="GANLoss", gan_type="vanilla", real_label_val=1.0, fake_label_val=1.0,
                           loss_weight=1.0),
    identity_loss=dict(identity_weight=10),
    )

# model training and testing settings
train_cfg = dict(fix_iter=5000, gfp_fix_iter=50000, r1_reg_weight=10,
                 net_d_iters=1, net_d_init_iters=0, net_d_reg_every= 16, world_size = 1)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# # dataset settings
# train_dataset_type = 'SSTERR_GANDataset'
# val_dataset_type = 'SSTERR_GANDataset'

# train_pipeline = [
#     dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:03d}.png'),
#     # dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
#     dict(
#         type='LoadImageFromFileList',
#         io_backend='disk',
#         key='lq',
#         channel_order='rgb'),
#     dict(
#         type='LoadImageFromFileList',
#         io_backend='disk',
#         key='gt',
#         channel_order='rgb'),
#     dict(type='Resize',keys=['lq', 'gt'], scale=(256, 256), keep_ratio=False),  
#     dict(type='LoadFacialComponent', component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/train_video.json'),
#     # Do hiện tại data có video < 256 nên khi chay random crop ở dưới bị lỗi nên tạm thời resize video lại
#     #Khi có data chuẩn >256 thì sửa lại code line 155 ở /home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/mmedit/datasets/pipelines/augmentation.py
#     dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
#     # dict(type='PairedRandomCrop', gt_patch_size=256),
#     # dict(
#     #     type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
#     #     direction='horizontal'),
#     # dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
#     # dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
#     dict(type='FramesToTensor', keys=['lq', 'gt']),
#     dict(type='Collect', keys=['lq', 'gt', 'facial_component'], meta_keys=['lq_path', 'gt_path'])
# ]

# test_pipeline = [
#     dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:03d}.png'),
#     dict(
#         type='LoadImageFromFileList',
#         io_backend='disk',
#         key='lq',
#         channel_order='rgb'),
#     dict(
#         type='LoadImageFromFileList',
#         io_backend='disk',
#         key='gt',
#         channel_order='rgb'),
#     dict(type='Resize',keys=['lq', 'gt'], scale=(256, 256), keep_ratio=False),
#     # dict(type='Resize',keys=['lq', 'gt'], scale=(256, 256), keep_ratio=False),
#     dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
#     dict(type='FramesToTensor', keys=['lq', 'gt']),
#     dict(
#         type='Collect',
#         keys=['lq', 'gt'],
#         meta_keys=['lq_path', 'gt_path', 'key'])
# ]

# demo_pipeline = [
#     dict(type='GenerateSegmentIndices', interval_list=[1]),
#     dict(
#         type='LoadImageFromFileList',
#         io_backend='disk',
#         key='lq',
#         channel_order='rgb'),
#     dict(type='RescaleToZeroOne', keys=['lq']),
#     dict(type='FramesToTensor', keys=['lq']),
#     dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
# ]

# data = dict(
#     workers_per_gpu=1,  #Need to be change when training
#     train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 2 gpus
#     val_dataloader=dict(samples_per_gpu=1),
#     test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

#     # train
#     train=dict(
#         type='RepeatDataset',
#         times=1000,
#         dataset=dict(
#             type=train_dataset_type,
#             lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/train/input',
#             gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/train/output',
#             component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/train_video.json',
#             num_input_frames=4,
#             pipeline=train_pipeline,
#             scale=1,
#             test_mode=False)),
#     # val
#     val=dict(
#         type=val_dataset_type,
#         lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/input',
#         gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/output',
#         component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/test_video.json',
#         num_input_frames=30,
#         pipeline=test_pipeline,
#         scale=1,
#         test_mode=True
#         ),
#     # test
#     test=dict(
#         type=val_dataset_type,
#         lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/input',
#         gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/output',
#         num_input_frames=30,
#         pipeline=test_pipeline,
#         scale=1,
#         # val_partition='REDS4',
#         test_mode=True),
# )

# # optimizer
# optimizers = dict(
#     generator=dict(
#         type='Adam',
#         lr=2e-3,
#         betas=(0.9, 0.99),
#         paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})),
#     # discriminator=dict(
#     #     type='Adam',
#     #     lr=2e-3),
#     # net_d_left_eye=dict(
#     #     type='Adam',
#     #     lr=2e-3),
#     # net_d_right_eye=dict(
#     #     type='Adam',
#     #     lr=2e-3),
#     # net_d_mouth=dict(
#     #     type='Adam',
#     #     lr=2e-3)
#     )

# # learning policy
# total_iters = 300000
# lr_config = dict(
#     policy='CosineRestart',
#     by_epoch=False,
#     periods=[300000],
#     restart_weights=[1],
#     min_lr=1e-7)

# checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# # remove gpu_collect=True in non distributed training
# evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
# log_config = dict(
#     interval=100,
#     hooks=[
#         dict(type='TextLoggerHook', by_epoch=False),
#         # dict(type='TensorboardLoggerHook'),
#     ])
# visual_config = None

# # runtime settings
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = f'./work_dirs/{exp_name}'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
# find_unused_parameters = True


# dataset settings
train_dataset_type = 'SSTERR_GANDataset'
val_dataset_type = 'SSTERR_GANDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:03d}.png'),
    # dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
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
    dict(type='Resize',keys=['lq', 'gt'], scale=(512, 512), keep_ratio=False),
    dict(type='LoadFacialComponent', component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/train_video.json'),
    # Do hiện tại data có video < 256 nên khi chay random crop ở dưới bị lỗi nên tạm thời resize video lại
    #Khi có data chuẩn >256 thì sửa lại code line 155 ở /home/ldtuan/VideoRestoration/BasicVSR_PlusPlus/mmedit/datasets/pipelines/augmentation.py
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    # dict(type='PairedRandomCrop', gt_patch_size=256),
    # dict(
    #     type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
    #     direction='horizontal'),
    # dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    # dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'facial_component'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], filename_tmpl='{:03d}.png'),
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
    dict(type='Resize',keys=['lq', 'gt'], scale=(512, 512), keep_ratio=False),
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
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    #train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/train/input',
            gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/train/output',
            component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/train_video.json',
            num_input_frames=4,
            pipeline=train_pipeline,
            scale=1,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/input',
        gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/output',
        component_file='/home/ldtuan/VideoRestoration/dataset/official_degradation/test_video.json',
        num_input_frames=30,
        pipeline=test_pipeline,
        scale=1,
        test_mode=True
        ),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/input',
        gt_folder='/home/ldtuan/VideoRestoration/dataset/official_degradation/test/output',
        num_input_frames=None,
        pipeline=test_pipeline,
        scale=1,
        # val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})),
    discriminator=dict(
        type='Adam',
        lr=2e-4),
    net_d_left_eye=dict(
        type='Adam',
        lr=2e-4),
    net_d_right_eye=dict(
        type='Adam',
        lr=2e-4),
    net_d_mouth=dict(
        type='Adam',
        lr=2e-4)
    )

# optimizer
# optimizers = dict(
#     generator=dict(
#         type='Adam',
#         lr=2e-3,
#         betas=(0.9, 0.99),
#         paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})),
#     discriminator=dict(
#         type='Adam',
#         lr=2e-3),
#     net_d_left_eye=dict(
#         type='Adam',
#         lr=2e-3),
#     net_d_right_eye=dict(
#         type='Adam',
#         lr=2e-3),
#     net_d_mouth=dict(
#         type='Adam',
#         lr=2e-3)
#     )

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
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
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