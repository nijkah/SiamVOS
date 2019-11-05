import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone
from models.aspp import build_aspp
from models.roi_utils import *

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

        

        #self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, -0.329]).view(1,4,1,1))
        #self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.051]).view(1,4,1,1))
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    #def fuse_mask(self, x, mask, is_ref=False):
    #    x = self.conv1_1(x)
    #    mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=True)
    #    fg = x * mask
    #    bg  = x - fg
    #    out = torch.cat([fg, bg], 1)
    #    out = self.fuse(out)
        #out = self.template_fuse(out) if is_ref else self.fuse(out)

    #    return out
        
    
    def extract_feature(self, x, mask, is_ref=False):
        mask = torch.cat([x, mask], 1)
        x = (x - self.mean) / self.std
        x = self.Scale.conv1(x) + self.conv1_rm(mask)
        x = self.Scale.bn1(x)
        ll = self.Scale.relu(x)
        low_level_feat = self.Scale.maxpool(ll)
        low_level_feat = self.Scale.layer1(low_level_feat)
        out = self.Scale.layer2(low_level_feat)
        out = self.Scale.layer3(out)
        out = self.Scale.layer4(out)

        return ll, low_level_feat, out

    def forward(self, x, mask, ref, box, gt=None):
        bn, _, oh, ow = ref.shape
        roi_of_ref = mask2yxhw(box, scale=2.0)
        ref_fw_grid, ref_bw_grid, theta = get_ROI_grid(roi_of_ref, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_ref = F.grid_sample(ref, ref_fw_grid)
        cropped_box = F.grid_sample(box, ref_fw_grid)
        
        _, _, template_feature = self.extract_feature(cropped_ref, cropped_box, is_ref=True)
        #template_feature = self.extract_feature(cropped_ref, cropped_box)

        oh, ow = x.shape[2:]
        roi = mask2yxhw(mask, scale=2.0)
        fw_grid, bw_grid, theta = get_ROI_grid(roi, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_x = F.grid_sample(x, fw_grid)
        cropped_mask = F.grid_sample(mask, fw_grid)

        ll, low_level_feat, out = self.extract_feature(cropped_x, cropped_mask)
        out = self.SEFA(out, template_feature)
        out_1 = self.aspp(out)

        #out = out + fused_feature
        #branch_feature = self.branch(low_level_feat)
        
        #mask = F.interpolate(mask, size=branch_feature.shape[2:])
        #mask_feature = branch_feature * mask
        #fused_feature = torch.cat([branch_feature, mask_feature], 1)
        #fused_feature = self.fuse(fused_feature)
        #branch_feature = branch_feature + fused_feature

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
        #b.append(self.template_fuse)

        
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

def build_siam_Deeplab(NoLabels=21, pretrained=False):
    model = Siam_Deeplab(NoLabels, pretrained)
    return model

