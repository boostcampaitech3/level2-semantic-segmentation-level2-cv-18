## 각 Architecture + backbone의 실험 결과

|Architecture|Backbone|mIoU|
|------|---|---|
|deeplabV3+|convNext|0.7131|
|deeplabV3+|resNet50|0.6317|
|knet|convNext|0.6610|
|ocrnet|hrnet|0.5495|
|UperNet|swin-large|0.6610|
|UperNet|convNext|0.7328|

## UperNet + convNext 모델의 실험 결과

|적용 기법|mIoU|  
|----------|------|
|Baseline|0.7328|
|Pseudo labeling|0.8023|
|Pseudo labeling+Augmentation|0.8030|
|Pseudo labeling+Augmentation+TTA|0.8044|
|Pseudo labeling+Augmentattion+copy&paste+TTA+seed 고정|0.8179|
