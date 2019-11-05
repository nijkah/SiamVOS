from models.backbone import resnet

def build_backbone(backbone, in_channel, layers='101', pretrained=True):
    ms = False
    if 'ms' in backbone:
        ms = True
    if 'resnet' in backbone:
        return resnet.build_resnet(in_channel, layers, ms, pretrained)

    return None
