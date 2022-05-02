_base_ = [
    '../models/fcn_hr18.py',
    # '../_base_/datasets/pascal_voc12_aug.py',
    # '../datasets/coco-stuff164k.py',
    '../datasets/custom_dataset.py',
    '../default_runtime.py',
    # '../schedules/schedule_20k.py'
    '../schedules/schedule.py'
]
model = dict(decode_head=dict(num_classes=11))
