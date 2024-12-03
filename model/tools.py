import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from util.utils import *


def build_cur_cls_label(mask, nclass):
    """some point annotations are cropped out, thus the prototypes are partial"""
    b = mask.size()[0]
    mask_one_hot = one_hot(mask, nclass)
    cur_cls_label = mask_one_hot.view(b, nclass, -1).max(-1)[0]
    return cur_cls_label.view(b, nclass, 1, 1)


def clean_mask(mask, cls_label, softmax=True):
    if softmax:
        mask = F.softmax(mask, dim=1)
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)


def get_cls_loss(predict, cls_label, mask):
    """cls_label: (b, k)"""
    """ predict: (b, k, h, w)"""
    """ mask: (b, h, w) """
    b, k, h, w = predict.size()
    predict = torch.softmax(predict, dim=1).view(b, k, -1)
    mask = mask.view(b, -1)

    # if a patch does not contain label k,
    # then none of the pixels in this patch can be assigned to label k
    loss = - (1 - cls_label.view(b, k, 1)) * torch.log(1 - predict + 1e-6)
    loss = torch.sum(loss, dim=1)
    loss = loss[mask != 255].mean()
    return loss


def one_hot(label, nclass):
    b, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(b, 1, h*w)

    mask = torch.zeros(b, nclass+1, h*w).to(label.device)
    mask = mask.scatter_(1, label_cp.long(), 1).view(b, nclass+1, h, w).float()
    return mask[:, :-1, :, :]


def one_hot_2d(label, nclass):
    h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, h*w)

    mask = torch.zeros(nclass+1, h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()
    return mask[:-1, :, :]


def cal_protypes(feat, mask, nclass):
    # print(f'feat.size(): {feat.size()}')    # [16, 256, 81, 81]
    # print(f'mask.size(): {mask.size()}')    # [16, 321, 321]
    feat = F.interpolate(feat, size=mask.size()[-2:], mode='bilinear')
    b, c, h, w = feat.size()
    # print(feat.size())  # [16, 256, 321, 321] 特征放到原图size
    prototypes = torch.zeros((b, nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)  # 计算每个图的每个类的原型，原型size为256（特征维度）
    for i in range(b):
        cur_mask = mask[i]  # 当前图的mask
        cur_mask_onehot = one_hot_2d(cur_mask, nclass)
        # print(f'cur_mask_onehot.size(): {cur_mask_onehot.shape}')  # [21, 321, 321]

        cur_feat = feat[i]  # 当前图的特征
        cur_prototype = torch.zeros((nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)  # 当前图的每个类的原型

        cur_set = list(torch.unique(cur_mask))  # 当前这个mask中存在的类或者背景或者255
        if nclass in cur_set:
            cur_set.remove(nclass)  # 删除背景
        if 255 in cur_set:
            cur_set.remove(255) # 删除255

        for cls in cur_set:
            m = cur_mask_onehot[cls].view(1, h, w)  # [1, 321, 321] 图中存在的类的mask
            sum = m.sum()   # 这个类的标签总共多少像素
            m = m.expand(c, h, w).view(c, -1)  # [256, 103041]  类标签有256个1，其他都是0

            cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1)/(sum + 1e-6)  # 当前图特征与上面mask相乘，取平均获得类别特征 256
            # print(f'cls_feat.size(): {cls_feat.shape}')
            # a = []
            # b = a[10]
            cur_prototype[cls, :] = cls_feat

        prototypes[i] += cur_prototype

    # 至此获得每个图的每个类的原型，原型size为256（特征维度）
    cur_cls_label = build_cur_cls_label(mask, nclass).view(b, nclass, 1)
    # print(f'cur_cls_label.size(): {cur_cls_label.shape}')   # [16, 21, 1]
    # a = []
    # b = a[10]

    mean_vecs = (prototypes.sum(0)*cur_cls_label.sum(0))/(cur_cls_label.sum(0)+1e-6)    # 对当前每个类都做平均，论文eq4
    # print(f'mean_vecs.size(): {mean_vecs.shape}')   # [21, 256]

    loss = proto_loss(prototypes, mean_vecs, cur_cls_label) # 论文eq10,增大不同高斯混合质心的距离

    return prototypes.view(b, nclass, c), loss


def proto_loss(prototypes, vecs, cur_cls_label):
    b, nclass, c = prototypes.size()

    # abs = torch.abs(prototypes - vecs).mean(2)
    # positive = torch.exp(-(abs * abs))
    # positive = (positive*cur_cls_label.view(b, nclass)).sum()/(cur_cls_label.sum()+1e-6)
    # positive_loss = 1 - positive

    vecs = vecs.view(nclass, c)
    total_cls_label = (cur_cls_label.sum(0) > 0).long()
    negative = torch.zeros(1,
                           dtype=prototypes.dtype,
                           device=prototypes.device)

    num = 0
    for i in range(nclass):
        if total_cls_label[i] == 1:
            for j in range(i+1, nclass):
                if total_cls_label[j] == 1:
                    if i != j:
                        num += 1
                        x, y = vecs[i].view(1, c), vecs[j].view(1, c)
                        abs = torch.abs(x - y).mean(1)
                        negative += torch.exp(-(abs * abs))
                        # print(negative)

    negative = negative/(num+1e-6)
    negative_loss = negative

    return negative_loss


def GMM(feat, vecs, pred, true_mask, cls_label):
    b, k, oh, ow = pred.size()
    # print(f'feat.size(): {feat.shape}')   # [16, 256, 81, 81]
    # print(f'vecs.size(): {vecs.shape}')   # [16, 21, 256]
    # print(f'pred.size(): {pred.shape}')   # [16, 21, 321, 321]
    # print(f'true_mask.size(): {true_mask.shape}')   # [16, 321, 321]
    # print(f'cls_label.size(): {cls_label.shape}')   # [16, 21, 1, 1]
    # a = []
    # b = a[10]

    preserve = (true_mask < 255).long().view(b, 1, oh, ow)  # 0为忽略像素，1为类别背景
    # print(f'preserve1: {torch.unique(preserve)}')
    # print(f'preserve1: {preserve.shape}')  # [16, 1, 321, 321]
    # a = []
    # b = a[10]
    preserve = F.interpolate(preserve.float(), size=feat.size()[-2:], mode='bilinear')
    # print(f'--preserve2: {torch.unique(preserve)}')
    # print(f'--preserve2: {preserve.shape}')  # [16, 1, 81, 81]
    # a = []
    # b = a[10]
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')  # [16, 21, 81, 81]
    _, _, h, w = pred.size()

    vecs = vecs.view(b, k, -1, 1, 1)
    # print(f'vecs: {vecs.shape}')  # [16, 21, 256, 1, 1] # 原型
    feat = feat.view(b, 1, -1, h, w)
    # print(f'feat: {feat.shape}')  # [16, 1, 256, 81, 81]
    # a = []
    # b = a[10]

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs)    # eq6
    # print(f'abs: {abs.shape}')  # [16, 21, 256, 81, 81]
    abs = torch.abs(feat - vecs).mean(2)    # eq5
    # print(f'abs: {abs.shape}')  # [16, 21, 81, 81]
    # a = []
    # b = a[10]

    abs = abs * cls_label.view(b, k, 1, 1) * preserve.view(b, 1, h, w)  # 保留对应的类别，不知道插值preserve的目的？原本的binary mask变成概率mask了。费解
    # abs2 = abs * cls_label.view(b, k, 1, 1)   # 好像应该在最后测试res是否相同，感觉* preserve没作用
    # print(f'abs: {abs.shape}')  # [16, 21, 81, 81]
    # print(abs2.equal(abs1))
    abs = abs.view(b, k, h*w)
    # print(f'abs: {abs.shape}')  # [16, 21, 6561]
    a = []
    b = a[10]

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, h, w)
    res = torch.exp(-(abs * abs))   # 相当于没有eq7中e的指数的分母部分了
    # res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))      # 这里好像是原文的写法，但是被作者注释了
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')    # 插值回原图尺寸
    res = res * cls_label.view(b, k, 1, 1)

    return res


def loss_calc(preds, label, ignore_index, reduction='mean', multi=False, class_weight=False,
              ohem=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label_cp = label.clone()
    label_cp[label == ignore_index] = 255

    if ohem:
        ce = OhemCrossEntropy(use_weight=True)
    else:
        if class_weight:
            weight = torch.FloatTensor(
                [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
                 1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
            ce = torch.nn.CrossEntropyLoss(
                ignore_index=255, reduction=reduction, weight=weight)
        else:
            ce = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    if multi:
        aux_pred, pred = preds
        loss = ce(aux_pred, label_cp.long())*0.4 + ce(pred, label_cp.long())*0.6

    else:
        loss = ce(preds, label_cp.long())
    return loss


def cal_gmm_loss(pred, res, cls_label, true_mask):
    n, k, h, w = pred.size()
    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)
    loss1 = loss1/2
    loss1 = (loss1*cls_label).sum(1)/(cls_label.sum(1)+1e-6)
    loss1 = loss1[true_mask != 255].mean()

    true_mask_one_hot = one_hot(true_mask, k)
    loss2 = - true_mask_one_hot * torch.log(res + 1e-6) \
            - (1 - true_mask_one_hot) * torch.log(1 - res + 1e-6)
    loss2 = loss2/2
    loss2 = (loss2 * cls_label).sum(1) / (cls_label.sum(1) + 1e-6)
    loss2 = loss2[true_mask < k].mean()
    return loss1+loss2


class OhemCrossEntropy(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=1e6, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # weight = torch.FloatTensor(
            #     [
            #         0.8373,
            #         0.918,
            #         0.866,
            #         1.0345,
            #         1.0166,
            #         0.9969,
            #         0.9754,
            #         1.0489,
            #         0.8786,
            #         1.0023,
            #         0.9539,
            #         0.9843,
            #         1.1116,
            #         0.9037,
            #         1.0865,
            #         1.0955,
            #         1.0865,
            #         1.1529,
            #         1.0507,
            #     ]
            # ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).to(label.device)
            weight = torch.FloatTensor(
                [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
                 1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


if __name__ == '__main__':
    proto = torch.rand(8, 21, 256)
    vecs = torch.rand(1, 21, 256)
    cls_label = torch.rand(8, 21, 1)
    proto_loss(proto, vecs, cls_label)
