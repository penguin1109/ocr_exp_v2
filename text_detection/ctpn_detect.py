"""ctpn_detect.py
1. clip_bboxes [X]
2. nms [X]
3. decode [ ]
4. TextDetector [ ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'src', 'utils'))
from ctpn_connector import TextProposalConnector
from ctpn_anchor import generate_all_anchor_boxes

""" (1) clip_boxes
- bounding box를 H, W, 0.0, 0.0에 대해서 내부 길이로 정해질 수 있도록 한다.
"""
def clip_boxes(bboxes, image_size):
    ## 원본 이미지의 크기의 가로, 세로보다 넘치거나 0보다 작은 길이를 가질수도 있어서 그부분 예외 처리
    H, W = image_size
    zero = 0.0
    W_diff, H_diff = W - 1., H - 1.
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], W_diff), zero)
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], H_diff), zero)

    return bboxes

def nms(bboxes, scores, iou_threshold):
    """ Non-Max Suppression 계산
    - bboxes: the bounding box coordinate
    - scores: scores for each bounding box
    Returns: a list containing the best indices out of a set of overlapping bbox
    """
    xmin, xmax = bboxes[:, 0], bboxes[:, 2]
    ymin, ymax = bboxes[:, 1], bboxes[:,3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    score_indices = np.argsort(scores, kind="mergesort", axis=-1)[::-1] ## 점수가 높은 bounding box부터 사용
    zero = 0.0
    candidates = []

    while score_indices.size > 0: ## 처음부터 score indices를 줄여 나가면서 x축 왼쪽 오른쪽
        # 그리고 y축 위 아래의 bounding box끝 위치를 계산한다.
        i = score_indices[0]
        candidates.append(i)
        xxmax = np.maximum(xmin[i], xmin[score_indices[1:]])
        yymax = np.maximum(ymin[i], ymin[score_indices[1:]])
        xxmin = np.minimum(xmax[i], xmax[score_indices[1:]])
        yymin = np.minimum(ymax[i], ymax[score_indices[1:]])

        W = np.maximum(zero, xxmin - xxmax)
        H = np.maximum(zero, yymin - yymax)

        overlap = W * H ## (교집합)
        remain = areas[score_indices[1:]] ## remaining areas (차집합)
        aou = areas[i] + remain - overlap ## area of union
        IoU = overlap / aou

        indices = np.where(IoU <= iou_threshold)[0]
        score_indices = score_indices[indices+1]
    
    return candidates ## bounding box의 좌표 자체가 아니라 후보 index를 보내주면
    # 나중에 후보군 anchor box들을 사용해서 text proposal connector을 사용해서 연결되는 
    # text line을 구할 수 있도록 한다.

def decode(predicted_bboxes, anchor_boxes):
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1. ## 원래 anchor box의 높이 
    anchor_center_y = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2. ## 원래 anchor box의 중심 y좌표
    truth_center_y = predicted_bboxes[..., 0] * anchor_height + anchor_center_y ## 예측된 bbox의 y축의 중심 좌표
    truth_height = np.exp(predicted_bboxes[..., 1]) * anchor_height ## 예측된 bbox의 높이

    x1 = anchor_boxes[:, 0] ## Min X (정해져 있는 anchor box의 왼쪽 x좌표를 의미)
    y1 = truth_center_y - truth_height / 2. ## Min Y
    x2 = anchor_boxes[:, 2] ## Max X (정해져 있는 anchor box의 오른쪽 x좌표를 의미)
    y2 = truth_center_y + truth_height / 2. ## Max Y
    
    bboxes = np.stack([x1, y1.squeeze(), x2, y2.squeeze()], axis = 1)

    return bboxes


class TextDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.CONF_SCORE = cfg.CONF_SCORE ## 0.9
        self.IOU_THRESH = cfg.IOU_THRESH ## 0.2
        self.FEATURE_STRIDE = cfg.FEATURE_STRIDE ## 16
        self.ANCHOR_SHIFT = cfg.ANCHOR_SHIFT ## 16
        self.text_proposal_connector = TextProposalConnector(self.cfg)

    def __call__(self, predictions, image_size):
        H, W = image_size
        predicted_bboxes, predicted_scores = predictions ## regr, cls
        predicted_scores = torch.softmax(predicted_scores,dim=2)
        predicted_bboxes = predicted_bboxes.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        feature_map_size = [int(np.ceil(H / self.ANCHOR_SHIFT)), int(np.ceil(W / self.ANCHOR_SHIFT))]
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size = feature_map_size,
            feature_stride = self.FEATURE_STRIDE,
            anchor_heights = self.cfg.ANCHOR_HEIGHTS,
            anchor_shift = self.ANCHOR_SHIFT
        )

        decoded_bboxes = decode(predicted_bboxes=predicted_bboxes, anchor_boxes=anchor_boxes)
        clipped_bboxes = clip_boxes(bboxes=decoded_bboxes, image_size=image_size)

        text_class = 1
        conf_scores = predicted_scores[0, :, text_class]
        conf_scores_mask = np.where(conf_scores > self.CONF_SCORE)[0] ## np.where을 사용해서 일정 점수 이상인 index의 값을 구해준다.
        
        selected_bboxes = clipped_bboxes[conf_scores_mask, :]
        selected_scores = predicted_scores[0, conf_scores_mask, text_class]
        
        candidates = nms(bboxes=selected_bboxes,
                    scores=selected_scores,
                    iou_threshold=self.IOU_THRESH
        )
        selected_bboxes, selected_scores = selected_bboxes[candidates], selected_scores[candidates]
        text_lines, scores = self.text_proposal_connector.get_text_lines(
            text_proposals=selected_bboxes,
            scores=selected_scores,
            image_size=image_size
        )

        return text_lines, scores


