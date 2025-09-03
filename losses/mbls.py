import torch
import torch.nn as nn
import torch.nn.functional as F

class V8DetectionLoss:
    def __init__(self, model, margin = None, mbls_alpha = None):

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

        print("using MbLS Loss")
        self.margin = getattr(model, "margin", margin if margin is not None else 1.0)
        self.mbls_alpha = getattr(model, "mbls_alpha", mbls_alpha if mbls_alpha is not None else 1.0)
        print(f"mbls alpha and margin are as follows: {self.mbls_alpha}, {self.margin}")
        # self.margin = 10
        # self.alpha = 0.1

    def get_diff(self, inputs):
        # print("dimension od the inputs: ", inputs.dim())
        max_values = inputs.max(dim=1)
        dims = inputs.shape
        max_vals = max_values.values.view(dims[0], *[1] * (inputs.dim() - 1)).repeat(1, *dims[1:])
        #max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_vals - inputs
        return diff
    
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

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # print("using MbLS main function")
        diff = self.get_diff(pred_scores)
        diff = diff.squeeze(-1)
        # print("Shape of diff: ", diff.shape)
        loss_margin = F.relu(diff - self.margin).mean()
        mbls_penalty = self.mbls_alpha * loss_margin
        loss[1]+= mbls_penalty