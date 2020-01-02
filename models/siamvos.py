import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone
from models.aspp import build_aspp
from models.roi_utils import *


class GC(nn.Module):
    def __init__(self, in_channel, out_channel, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channel, 256, kernel_size=(kh, 1),
                                padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, out_channel, kernel_size=(1, kw),
                                padding=(0, int(kw//2)))
        self.conv_r1 = nn.Conv2d(in_channel, 256, kernel_size=(1, kw),
                                padding=(0, int(kh/2)))
        self.conv_r2 = nn.Conv2d(256, out_channel, kernel_size=(kh, 1),
                                padding=(int(kh/2), 0))
    def forward(self, out, out2):
        x = torch.cat([out, out2], dim=1)
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x

class SEFA(nn.Module):
    def __init__(self, inplanes, r=4):
        super(SEFA, self).__init__()
        self.inplanes = inplanes
        self.fc1 = nn.Linear(2*inplanes, int(2*inplanes/r))
        self.fc2 = nn.Linear(int(2*inplanes/r), 2*inplanes)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = x.mean(-1).mean(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x.view(-1, self.inplanes, 2), dim=2)
        w1 = x[:,:,0].contiguous().view(-1, self.inplanes, 1, 1)
        w2 = x[:,:,1].contiguous().view(-1, self.inplanes, 1, 1)
        out = x1*w1 + x2*w2
        return out

class Siam_Deeplab_GC(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(Siam_Deeplab_GC, self).__init__()
        self.Scale = build_backbone('resnet', in_channel=3, pretrained=pretrained)
        self.aspp = build_aspp(output_stride=16)

        self.GC = GC(4096, 2048)

        self.conv1_rm = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=True)

        self.conv_low_1 = nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU())

        self.refine= nn.Sequential(
                nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.1))

        self.predict = nn.Sequential(
                nn.Conv2d(256+64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, NoLabels, kernel_size=1))

        

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_feature(self, x, mask, x_prev=None, is_ref=False):
        x = (x - self.mean) / self.std
        if x_prev is not None:
            x_prev = (x_prev - self.mean) / self.std
            mask = torch.cat([x_prev, mask], 1)
        else:
            mask = torch.cat([x, mask], 1)
        x = self.Scale.conv1(x) + self.conv1_rm(mask)
        x = self.Scale.bn1(x)
        ll = self.Scale.relu(x)
        low_level_feat = self.Scale.maxpool(ll)
        low_level_feat = self.Scale.layer1(low_level_feat)
        out = self.Scale.layer2(low_level_feat)
        out = self.Scale.layer3(out)
        out = self.Scale.layer4(out)

        return ll, low_level_feat, out

    def forward(self, x, mask, x_prev, ref, box, gt=None):
        bn, _, oh, ow = ref.shape
        roi_of_ref = mask2yxhw(box, scale=2.0)
        ref_fw_grid, ref_bw_grid, theta = get_ROI_grid(roi_of_ref, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_ref = F.grid_sample(ref, ref_fw_grid)
        cropped_box = F.grid_sample(box, ref_fw_grid)
        
        _, _, template_feature = self.extract_feature(cropped_ref, cropped_box, is_ref=True)

        oh, ow = x.shape[2:]
        roi = mask2yxhw(mask, scale=2.0)
        fw_grid, bw_grid, theta = get_ROI_grid(roi, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_x = F.grid_sample(x, fw_grid)
        cropped_x_prev = F.grid_sample(x_prev, fw_grid)
        cropped_mask = F.grid_sample(mask, fw_grid)

        ll, low_level_feat, out = self.extract_feature(cropped_x, cropped_mask, cropped_x_prev)
        out = self.GC(out, template_feature)
        out_1 = self.aspp(out)

        out = F.interpolate(out_1, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv_low_1(low_level_feat)
        out = torch.cat([out, low_level_feat], 1)
        out_2 = self.refine(out)
        out = F.interpolate(out_2, size=ll.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, ll], 1)
        out = self.predict(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        # Losses are computed within ROI
        if self.training:
            cropped_gt = F.grid_sample(gt, fw_grid)
            cropped_gt = cropped_gt.detach()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, torch.round(cropped_gt).long().view(bn, 256, 256))
        else:
            loss = 0

        out = F.grid_sample(out, bw_grid, padding_mode='border')
        
        return out, loss

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.Scale.conv1)
        b.append(self.Scale.bn1)
        b.append(self.Scale.layer1)
        b.append(self.Scale.layer2)
        b.append(self.Scale.layer3)
        b.append(self.Scale.layer4)
        b.append(self.conv_low_1)
        b.append(self.conv1_rm)
        #b.append(self.SEFA)
        #b.append(self.template_fuse)
        for i in self.GC.modules():
            for j in i.parameters():
                if j.requires_grad:
                    yield j

        
        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(self.aspp)
        b.append(self.refine)
        b.append(self.predict)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

class Siam_Deeplab(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(Siam_Deeplab, self).__init__()
        self.Scale = build_backbone('resnet', in_channel=3, pretrained=pretrained)
        self.aspp = build_aspp(output_stride=16)

        self.SEFA = SEFA(2048)

        self.conv1_rm = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=True)

        self.conv_low_1 = nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU())

        self.refine= nn.Sequential(
                nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.1))

        self.predict = nn.Sequential(
                #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(256+64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, NoLabels, kernel_size=1))

        

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def extract_feature(self, x, mask, x_prev=None, is_ref=False):
        x = (x - self.mean) / self.std
        mask = torch.cat([x, mask], 1)
        x = self.Scale.conv1(x) + self.conv1_rm(mask)
        x = self.Scale.bn1(x)
        ll = self.Scale.relu(x)
        low_level_feat = self.Scale.maxpool(ll)
        low_level_feat = self.Scale.layer1(low_level_feat)
        out = self.Scale.layer2(low_level_feat)
        out = self.Scale.layer3(out)
        out = self.Scale.layer4(out)

        return ll, low_level_feat, out

    def forward(self, x, mask, x_prev, ref, box, gt=None):
        bn, _, oh, ow = ref.shape
        roi_of_ref = mask2yxhw(box, scale=2.0)
        ref_fw_grid, ref_bw_grid, theta = get_ROI_grid(roi_of_ref, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_ref = F.grid_sample(ref, ref_fw_grid)
        cropped_box = F.grid_sample(box, ref_fw_grid)
        
        _, _, template_feature = self.extract_feature(cropped_ref, cropped_box, is_ref=True)

        oh, ow = x.shape[2:]
        roi = mask2yxhw(mask, scale=2.0)
        fw_grid, bw_grid, theta = get_ROI_grid(roi, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_x = F.grid_sample(x, fw_grid)
        cropped_x_prev = F.grid_sample(x_prev, fw_grid)
        cropped_mask = F.grid_sample(mask, fw_grid)

        ll, low_level_feat, out = self.extract_feature(cropped_x, cropped_mask, cropped_x_prev)
        out = self.SEFA(out, template_feature)
        out_1 = self.aspp(out)
       
        out = F.interpolate(out_1, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv_low_1(low_level_feat)
        out = torch.cat([out, low_level_feat], 1)
        out_2 = self.refine(out)
        out = F.interpolate(out_2, size=ll.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, ll], 1)
        out = self.predict(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        # Losses are computed within ROI
        if self.training:
            cropped_gt = F.grid_sample(gt, fw_grid)
            cropped_gt = cropped_gt.detach()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, torch.round(cropped_gt).long().view(bn, 256, 256))
        else:
            loss = 0

        out = F.grid_sample(out, bw_grid)
        
        return out, loss

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.Scale.conv1)
        b.append(self.Scale.bn1)
        b.append(self.Scale.layer1)
        b.append(self.Scale.layer2)
        b.append(self.Scale.layer3)
        b.append(self.Scale.layer4)
        b.append(self.conv_low_1)
        b.append(self.conv1_rm)
        b.append(self.SEFA)
        
        
        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(self.aspp)
        b.append(self.refine)
        b.append(self.predict)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

def build_siamvos(NoLabels=2,gc=False, pretrained=False):
    if gc:
        model = Siam_Deeplab_GC(NoLabels, pretrained)
    else:
        model = Siam_Deeplab(NoLabels, pretrained)
    return model

