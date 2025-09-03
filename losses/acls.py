import torch
import torch.nn as nn
import torch.nn.functional as F

class V8DetectionLoss:
    def __init__(self, model):

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        

        self.pos_lambda = 1.0
        self.neg_lambda = 0.1
        self.margin = 10 
        self.alpha = 0.1
        
        self.beta = 0.5
        print("positive lambda: ", self.pos_lambda)
        print("negative lambda: ", self.neg_lambda)
        print("acls alpha: ", self.acls_alpha)
        print("margin acls: ", self.margin)
        print("beta: ", self.beta)
        
def get_reg(self, cls_logits, cls_targets):

        # FOR OBJECT DETECTION
        max_values, pred_indices = cls_logits.max(dim=1, keepdim=True)  # (N, 1)
        max_values = max_values.repeat(1, cls_logits.size(1))  # (N, C)
        
        # Indicator: 1 for correct class, 0 for others
        indicator = torch.zeros_like(cls_logits)
        indicator.scatter_(1, cls_targets.unsqueeze(1), 1.0)

        # Distances
        neg_dist = max_values.detach() - cls_logits
        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        # Losses
        pos = indicator * pos_dist_margin**2
        neg = (1.0 - indicator) * (neg_dist_margin**2)

        num_pos = indicator.sum()
        num_neg = (1.0 - indicator).sum()

        # Avoid division by zero
        num_pos = torch.clamp(num_pos, min=1.0)
        num_neg = torch.clamp(num_neg, min=1.0)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg

def __call__(self, preds, batch):

    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
    # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
    # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)


    fg_mask = fg_mask.bool()
    cls_logits = pred_scores[fg_mask]           # Shape: (N, C), raw logits
    cls_targets = target_scores[fg_mask].argmax(dim=1)  

    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    loss_reg = self.get_reg(cls_logits, cls_targets)
    acls_penaty = self.acls_alpha * loss_reg
    loss[1]+= acls_penaty