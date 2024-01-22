import torch
import torch.nn as nn


def binary_cross_entropy_with_iou(preds, labels):
    # Binary Cross-Entropy Loss
    bce_loss = nn.BCELoss()
    probabilities= torch.sigmoid(preds)

    c_loss = bce_loss(probabilities, labels) * 0.5
    ground_truth = torch.where(labels == 1, torch.tensor(1.0), torch.tensor(0.0))
    # Intersection over Union
    intersection = (probabilities * ground_truth).sum()

    total = (probabilities + labels).sum()


    union = total - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    loss = c_loss-(torch.log(iou))*0.5

    return loss

def multiclass_cross_entropy_with_iou(preds, labels, num_classes):
    # Multiclass Cross-Entropy Loss
    ce_loss = nn.NLLLoss()

    loss = 0.5*ce_loss(preds, labels)

    # IoU for each class
    ious = []
    for cls in range(num_classes):
        
        label_cls = (labels == cls).float
        pred_cls = torch.exp(preds[:, cls])
        intersection = (label_cls * pred_cls).float().sum()
        union = (pred_cls.sum() + label_cls.sum()) - intersection

       
        ious.append((intersection + 1e-6) / (union + 1e-6))
    iou = ious.sum()

    loss = loss -torch.log(iou)*0.5

    return loss