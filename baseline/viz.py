import argparse
import json
import os
from itertools import chain
from PIL import ImageOps, Image, ImageDraw
from typing import Sequence, Tuple

import plotly.express as px
from plotly import subplots
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
from pycocotools.coco import COCO
import cv2
import numpy as np
import pandas as pd


# -- argparse
parser = argparse.ArgumentParser(description='Dataset Visualization')
parser.add_argument(
    '--root_dir',
    type=str,
    default='/opt/ml/input/data',
    help='데이터 루트 디렉토리',
)
parser.add_argument(
    '--annotation_file_name', 
    type=str, 
    default='train_all', 
    help='어노테이션 파일 명 (.json 생략)',
)
parser.add_argument(
    '--port',
    type=int,
    default=6006,
    help='dash app을 올릴 포트',
)

args = parser.parse_args()


# --
point = Tuple[int, int]

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def read_img(idx: int) -> Image:
    """이미지 로드 후 텍스트 영역 폴리곤을 표시하여 반환한다."""
    # load image, annotation
    image_id = coco.getImgIds()[idx]
    image_infos = coco.loadImgs(image_id)[0]
    img_path = os.path.join(args.root_dir, image_infos['file_name'])

    img = Image.open(img_path)

    ann_ids = coco.getAnnIds(imgIds=image_infos['id'])
    anns = coco.loadAnns(ann_ids)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    masks = np.zeros((image_infos["height"], image_infos["width"]))
    # General trash = 1, ... , Cigarette = 10
    anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
    for i in range(len(anns)):
        className = get_classname(anns[i]['category_id'], cats)
        pixel_value = category_names.index(className)
        masks[coco.annToMask(anns[i]) == 1] = pixel_value
    masks = masks.astype(np.int8)

    return img, masks, image_infos


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """

    class_colormap = pd.read_csv("class_dict.csv")
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


# -- load annotations
with open(os.path.join(args.root_dir, args.annotation_file_name)+'.json', 'r') as f:
    anno = json.load(f)

data_dir = os.path.join(args.root_dir, args.annotation_file_name)+'.json'
coco = COCO(data_dir)

num_files = len(coco.getImgIds())

img_idx = -1

color_img = Image.open('./color.png')

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H3(id='img_name'),
        html.Button('prev', id='btn-prev', n_clicks=0),
        html.Button('next', id='btn-next', n_clicks=0),
        dcc.Input(
            id='idx_input',
            type='number', 
            placeholder='input image index', 
            min=0, 
            max=num_files-1,
        ),
        html.Button('go', id='btn-go', n_clicks=0),
        dcc.Graph(id='graph'),
    ]
)

@app.callback(
    Output('graph', 'figure'),
    Output('img_name', 'children'),
    Output('idx_input', 'value'),
    Input('btn-next', 'n_clicks'),
    Input('btn-prev', 'n_clicks'),
    Input('btn-go', 'n_clicks'),
    State('idx_input', 'value')
)
def update_img(btn_n, btn_p, btn_g, go_idx: int):
    """버튼 이벤트에 반응, 이미지 업데이트"""
    global img_idx
    change_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'btn-next' in change_id:  # 다음
        img_idx = (img_idx + 1) % num_files
    elif 'btn-prev' in change_id:  # 이전
        img_idx = num_files-1 if img_idx == 0 else img_idx - 1
    elif 'btn-go' in change_id:  # 특정 인덱스로 이동
        img_idx = int(go_idx)
    else:  # 초기화
        img_idx = 0

    # img_path = os.path.join(args.root_dir, fnames[img_idx])
    img, mask, image_info = read_img(img_idx)
    fig = subplots.make_subplots(rows=1, cols=3)
    fig.add_trace(px.imshow(img).data[0], row=1, col=1)
    fig.add_trace(px.imshow(label_to_color_image(mask)).data[0], row=1, col=2)
    fig.add_trace(px.imshow(color_img).data[0], row=1, col=3)

    return fig, f'[{img_idx+1}/{num_files}] ' + image_info['file_name'], img_idx


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=args.port)