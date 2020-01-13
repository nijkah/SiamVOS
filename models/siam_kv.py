import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone
from models.aspp import build_aspp
from models.roi_utils import *
from models.stm import STM_head


class Decoder(nn.Module):
    def __init__(self, NoLabels, feature_dim, pretrained=False):
        super(Decoder, self).__init__()

        self.conv_low_1 = nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU())

        self.refine = nn.Sequential(
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

    def forward(self, ll, low_level_feat, out):
        out = F.interpolate(out, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv_low_1(low_level_feat)
        out = torch.cat([out, low_level_feat], 1)
        out_2 = self.refine(out)
        out = F.interpolate(out_2, size=ll.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([out, ll], 1)
        out = self.predict(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out

class Siam_kv(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(Siam_kv, self).__init__()
        self.backbone_m = build_backbone('resnet', in_channel=3, layers='50', pretrained=pretrained)
        self.backbone_q = build_backbone('resnet', in_channel=3, layers='50', pretrained=pretrained)
        self.aspp = build_aspp(output_stride=16)
        self.stm_head = STM_head()
        self.decoder = Decoder(NoLabels=NoLabels, feature_dim=2048)

        self.wg_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv_key= nn.Conv2d(2048, 2048//8, kernel_size=3, padding=1)
        self.conv_val = nn.Conv2d(2048, 2048//2, kernel_size=3, padding=1)

        #self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, -0.329]).view(1,4,1,1))
        #self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.051]).view(1,4,1,1))
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        torch.nn.init.kaiming_normal_(self.wg_conv.weight)
        torch.nn.init.kaiming_normal_(self.conv_key.weight)
        torch.nn.init.kaiming_normal_(self.conv_val.weight)
       
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def extract_feature_q(self, x):
        x = (x - self.mean) / self.std
        x = self.backbone_q.conv1(x)
        x = self.backbone_q.bn1(x)
        ll = self.backbone_q.relu(x)
        low_level_feat = self.backbone_q.maxpool(ll)
        low_level_feat = self.backbone_q.layer1(low_level_feat)
        out = self.backbone_q.layer2(low_level_feat)
        out = self.backbone_q.layer3(out)
        out = self.backbone_q.layer4(out)

        return ll, low_level_feat, out

    def extract_feature_m(self, x, mask):
        x = (x - self.mean) / self.std
        mask = torch.cat([x, mask], 1)
        x = self.backbone_m.conv1(x) + self.wg_conv(mask)

        x = self.backbone_m.bn1(x)
        ll = self.backbone_m.relu(x)
        low_level_feat = self.backbone_m.maxpool(ll)
        low_level_feat = self.backbone_m.layer1(low_level_feat)
        out = self.backbone_m.layer2(low_level_feat)
        out = self.backbone_m.layer3(out)
        out = self.backbone_m.layer4(out)

        return out


    def extract_kv(self, f,is_q=False):
        key = self.conv_key(f)
        val = self.conv_val(f)

        return key, val

    def forward(self, x, mask, x_prev, ref, box, gt=None):
        bn, _, oh, ow = ref.shape
        roi_of_ref = mask2yxhw(box, scale=2.0)
        ref_fw_grid, ref_bw_grid, theta = get_ROI_grid(roi_of_ref, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_ref = F.grid_sample(ref, ref_fw_grid)
        cropped_box = F.grid_sample(box, ref_fw_grid)
        
        ref_feature = self.extract_feature_m(cropped_ref, cropped_box)
        ref_key, ref_val = self.extract_kv(ref_feature)

        oh, ow = x.shape[2:]
        roi = mask2yxhw(mask, scale=2.0)
        fw_grid, bw_grid, theta = get_ROI_grid(roi, src_size=(oh, ow), dst_size=(256, 256), scale=1.0)
        cropped_x = F.grid_sample(x, fw_grid)
        cropped_x_prev = F.grid_sample(x_prev, fw_grid)
        cropped_mask = F.grid_sample(mask, fw_grid)

        prev_feature = self.extract_feature_m(cropped_x_prev, cropped_mask)
        prev_key, prev_val = self.extract_kv(prev_feature)
        mem_key = torch.cat([ref_key.unsqueeze(2), prev_key.unsqueeze(2)], 2)
        mem_val = torch.cat([ref_val.unsqueeze(2), prev_val.unsqueeze(2)], 2)

        ll, low_level_feat, out = self.extract_feature_q(cropped_x)
        out_key, out_val = self.extract_kv(out, is_q=True)

        out = self.stm_head(out_key, out_val, mem_key, mem_val)
        out_1 = self.aspp(out)

        out = self.decoder(ll, low_level_feat, out_1)

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

        b.append(self.backbone_q.conv1)
        b.append(self.backbone_q.bn1)
        b.append(self.backbone_q.layer1)
        b.append(self.backbone_q.layer2)
        b.append(self.backbone_q.layer3)
        b.append(self.backbone_q.layer4)
        b.append(self.backbone_m.conv1)
        b.append(self.backbone_m.bn1)
        b.append(self.backbone_m.layer1)
        b.append(self.backbone_m.layer2)
        b.append(self.backbone_m.layer3)
        b.append(self.backbone_m.layer4)
        b.append(self.wg_conv)
        b.append(self.conv_key)
        b.append(self.conv_val)

        
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
        b.append(self.decoder.conv_low_1)
        b.append(self.decoder.refine)
        b.append(self.decoder.predict)

        for i in range(len(b)):
            for j in b[i].modules():
                for k in j.parameters():
                    if k.requires_grad:
                        yield k

def build_siam_Deeplab(NoLabels=21, pretrained=False):
    model = Siam_kv(NoLabels, pretrained)
    return model

