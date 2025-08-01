import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=0.25):
        """
        Focal Loss class for multi-class segmentation.
        :param num_classes (required): Number of classes (only required for multi-class classification)
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        """
        super().__init__()
        assert isinstance(alpha, int) or isinstance(alpha, float) or isinstance(alpha, list) # Ensure 'alpha' contains valid values.

        self.num_classes = num_classes
        self.gamma = gamma
        if isinstance(alpha, list):
          self.alpha = torch.Tensor(alpha)
        else:
          self.alpha = alpha


    def forward(self, inputs, targets):
        inputs = inputs.reshape(0, 2, 3, 1).view(-1, inputs.shape[3])
        targets = targets.view(-1)
        return self.multi_class_focal_loss(inputs, targets)


    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class segmentation. """
        alpha = self.alpha.to(inputs.device)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        ce_loss = -targets_one_hot * torch.log(probs)

        if isinstance(self.alpha, list):
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        return (focal_weight.unsqueeze(1) * ce_loss).mean()
