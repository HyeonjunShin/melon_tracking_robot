import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        # chunk 대신 split을 사용하여 FX Tracer 안정성 확보
        y1, y2 = torch.split(self.cv1(x), self.c, dim=1)
        y = [y1, y2]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class StandardYOLOv8Head(nn.Module):
    def __init__(self, ch_channels=[64, 128, 256], num_classes=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.reg_output_dim = 4 * reg_max

        self.cls_cvs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, self.num_classes, 1)
            )
            for c in ch_channels
        )
        self.reg_cvs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, self.reg_output_dim, 1)
            )
            for c in ch_channels
        )

    def forward(self, feats):
        cls_outputs = []
        reg_outputs = []
        for i, x in enumerate(feats):
            cls_logits = self.cls_cvs[i](x)  # [B, 1, H, W]
            reg_dist = self.reg_cvs[i](x)  # [B, 64, H, W]

            # LiteRT / TFLite 친화적인 Reshape 사용 (Dynamic Shape 호환)
            b = cls_logits.shape[0]
            cls_outputs.append(cls_logits.reshape(b, self.num_classes, -1))
            reg_outputs.append(reg_dist.reshape(b, self.reg_output_dim, -1))

        cls_all = torch.cat(cls_outputs, dim=-1).permute(0, 2, 1).contiguous()
        reg_all = torch.cat(reg_outputs, dim=-1).permute(0, 2, 1).contiguous()
        return cls_all, reg_all


class DetectionModel(nn.Module):
    def __init__(self, num_classes=1, reg_max=16):
        super().__init__()

        self.stem = Conv(3, 16, 3, 2)
        self.bl1 = C2f(16, 32, n=1, shortcut=True)

        self.down1 = Conv(32, 64, 3, 2)
        self.bl2 = C2f(64, 64, n=2, shortcut=True)

        self.down2 = Conv(64, 128, 3, 2)
        self.bl3 = C2f(128, 128, n=2, shortcut=True)

        self.down3 = Conv(128, 256, 3, 2)
        self.bl4 = C2f(256, 256, n=1, shortcut=True)

        self.down4 = Conv(256, 256, 3, 2)
        self.sppf = SPPF(256, 256, k=5)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv_p4_fuse = Conv(256 + 256, 256, 1, 1)
        self.c2f_p4_up = C2f(256, 128, n=1, shortcut=False)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv_p3_fuse = Conv(128 + 128, 128, 1, 1)
        self.c2f_p3_up = C2f(128, 64, n=1, shortcut=False)

        self.down_p3 = Conv(64, 64, 3, 2)
        self.cv_p4_down = Conv(64 + 128, 128, 1, 1)
        self.c2f_p4_down = C2f(128, 128, n=1, shortcut=False)

        self.down_p4 = Conv(128, 128, 3, 2)
        self.cv_p5_down = Conv(128 + 256, 256, 1, 1)
        self.c2f_p5_down = C2f(256, 256, n=1, shortcut=False)

        self.one2many_head = StandardYOLOv8Head(
            ch_channels=[64, 128, 256], num_classes=num_classes, reg_max=reg_max
        )
        self.one2one_head = StandardYOLOv8Head(
            ch_channels=[64, 128, 256], num_classes=num_classes, reg_max=reg_max
        )

    def forward(self, x):
        # Backbone
        x_stem = self.stem(x)
        x_stage2 = self.bl2(self.down1(self.bl1(x_stem)))
        x_p3_raw = self.bl3(self.down2(x_stage2))
        x_p4_raw = self.bl4(self.down3(x_p3_raw))
        x_p5_raw = self.sppf(self.down4(x_p4_raw))

        # PAFPN Neck
        p5_up = self.up1(x_p5_raw)
        p4_fuse = self.cv_p4_fuse(torch.cat([p5_up, x_p4_raw], dim=1))
        p4_up_feat = self.c2f_p4_up(p4_fuse)

        p4_up = self.up2(p4_up_feat)
        p3_fuse = self.cv_p3_fuse(torch.cat([p4_up, x_p3_raw], dim=1))
        p3_out = self.c2f_p3_up(p3_fuse)

        p3_down = self.down_p3(p3_out)
        p4_down_fuse = self.cv_p4_down(torch.cat([p3_down, p4_up_feat], dim=1))
        p4_out = self.c2f_p4_down(p4_down_fuse)

        p4_down = self.down_p4(p4_out)
        p5_down_fuse = self.cv_p5_down(torch.cat([p4_down, x_p5_raw], dim=1))
        p5_out = self.c2f_p5_down(p5_down_fuse)

        final_features = [p3_out, p4_out, p5_out]

        if self.training:
            return self.one2many_head(final_features), self.one2one_head(final_features)
        else:
            return self.one2one_head(final_features)
class DecoderModule(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        
        CONF = [(8, 48, 80), (16, 24, 40), (32, 12, 20)]
        
        anchors_list = []
        strides_list = []
        
        for stride, h, w in CONF:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32), 
                torch.arange(w, dtype=torch.float32), 
                indexing="ij"
            )
            
            grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2) + 0.5
            anchors_list.append(grid)
            
            strides_list.append(torch.full((grid.size(0), 1), stride, dtype=torch.float32))
            
        anchors = torch.cat(anchors_list, dim=0)
        strides = torch.cat(strides_list, dim=0)
        
        self.register_buffer("anchors", anchors)
        self.register_buffer("strides", strides)
        self.register_buffer("weights", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, cls_pred, reg_pred):
        batch_size = reg_pred.size(0)
        
        reg_pred = reg_pred.view(batch_size, -1, 4, self.reg_max)
        
        softmax_reg = torch.softmax(reg_pred, dim=-1)
        dist = torch.sum(softmax_reg * self.weights, dim=-1)
        
        x1 = (self.anchors[:, 0] - dist[..., 0]) * self.strides[:, 0]
        y1 = (self.anchors[:, 1] - dist[..., 1]) * self.strides[:, 0]
        x2 = (self.anchors[:, 0] + dist[..., 2]) * self.strides[:, 0]
        y2 = (self.anchors[:, 1] + dist[..., 3]) * self.strides[:, 0]
        
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        cls_scores = torch.sigmoid(cls_pred)
        
        return bboxes, cls_scores

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = DetectionModel()
        self.decoder = DecoderModule() 
        
    def forward(self, x):
        cls_pred, reg_pred = self.base_model(x)
        return self.decoder(cls_pred, reg_pred)
    
if __name__ == "__main__":
    dummy = torch.randn(1, 3, 384, 640)  # 변환 시 Batch Size는 1 권장
    model = CombinedModel().eval()  # 변환 시에는 eval() 상태 사용

    out = model(dummy)
    print("⚡ [검증 성공] 출력 텐서 Shape:", out[0].shape, out[1].shape)


