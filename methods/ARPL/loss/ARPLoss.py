import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.ARPL.loss.Dist import Dist
from methods.ARPL.loss.LabelSmoothing import smooth_cross_entropy_loss

class ARPLoss(nn.CrossEntropyLoss):

    def __init__(self, **options):

        super(ARPLoss, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

        # Add label smoothing
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):
        dist_dot_p = self.Dist(x, center=self.points, metric='dot')
        dist_l2_p = self.Dist(x, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None: return logits, 0

        # Option for label smoothing
        if not self.label_smoothing:
            loss = F.cross_entropy(logits / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        loss = loss + self.weight_pl * loss_r

        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
