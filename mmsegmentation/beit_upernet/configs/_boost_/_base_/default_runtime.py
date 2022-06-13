# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', interval=50),
        # dict(type='WandbLoggerHook',interval=10,
        # init_kwargs=dict(
        #     project = 'beit_upernet',
        #     entity = 'cv18',
        #     name = 'beit_L_upernet_224_22k'
        # ),)
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
