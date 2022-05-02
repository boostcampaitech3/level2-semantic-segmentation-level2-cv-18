_base_ = [
    '../models/fcn_hr18.py',
    '../datasets/custom_dataset.py',
    '../default_runtime.py',
    '../schedules/schedule.py'
]
model = dict(decode_head=dict(num_classes=11))
