# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', interval=1000,
            init_kwargs=dict(
                project='knet_swin-l_pseudo',
                entity = 'cv18',
                name = '8148'
            ),
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/input/code/mmsegmentation/work_dirs/knet_s3_upernet_swin-l_pseudo2/best_mIoU_epoch_10_1.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
