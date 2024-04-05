import torch
import numpy as np


def center_to_corners_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    corners_bboxes = torch.zeros_like(bboxes)
    corners_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min
    corners_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2  # x_max
    corners_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min
    corners_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2  # y_max
    return corners_bboxes


def coco_to_corners_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    corners_bboxes = torch.zeros_like(bboxes)
    corners_bboxes[:, 0] = bboxes[:, 0]  # x_min stays the same
    corners_bboxes[:, 1] = bboxes[:, 1]  # y_min stays the same
    corners_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max = x_min + width
    corners_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max = y_min + height
    return corners_bboxes


def coco_to_center_bbox(bboxes):
    # Assuming bboxes is a torch tensor of shape [N, 4]
    center_bboxes = torch.zeros_like(bboxes)
    center_bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # center_x=x_min+width/2
    center_bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # center_y =y_min+height/2
    center_bboxes[:, 2] = bboxes[:, 2]  # width stays the same
    center_bboxes[:, 3] = bboxes[:, 3]  # height stays the same
    return center_bboxes


def scale_anchors(unit_anchors, width, height):
    anchors = unit_anchors.copy()
    anchors[:[0, 2]] *= width
    anchors[:[1, 3]] *= height
    return anchors


def generate_anchores(image_size, scales, aspect_ratios, feature_map_size):
    anchor_boxes_centers = []  # For center coordinates
    anchor_boxes_corners = []  # For corner coordinates

    step_size = image_size / feature_map_size  # Size of one grid cell

    for x in range(feature_map_size):
        for y in range(feature_map_size):
            center_x = (x + 0.5) * step_size
            center_y = (y + 0.5) * step_size

            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    box_height = image_size * scale / np.sqrt(aspect_ratio)
                    box_width = image_size * scale * np.sqrt(aspect_ratio)

                    # Center coordinates with width and height
                    anchor_boxes_centers.append(
                        [center_x, center_y, box_width, box_height]
                    )

                    # Convert to corner coordinates
                    x_min = center_x - box_width / 2
                    y_min = center_y - box_height / 2
                    x_max = center_x + box_width / 2
                    y_max = center_y + box_height / 2
                    anchor_boxes_corners.append([x_min, y_min, x_max, y_max])

    return np.array(anchor_boxes_centers), np.array(anchor_boxes_corners)
