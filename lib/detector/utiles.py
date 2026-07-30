import torch
import torch.nn.functional as F


class BBoxDecoder:
    def __init__(self, conf, num_bboxes=5040, reg_max=16, device=torch.device("cuda")):
        self.num_bboxes = num_bboxes
        self.reg_max = reg_max
        self.device = device

        self.grids = []
        self.strids = []

        for strid, height, width in conf:
            shift_y, shift_x = torch.meshgrid(
                torch.arange(height, device=device) + 0.5,
                torch.arange(width, device=device) + 0.5,
                indexing="ij",
            )
            anchor_points = torch.stack([shift_x, shift_y], dim=-1).view(-1, 2) * strid
            strid_tensor = torch.full((anchor_points.shape[0], 1), strid, device=device)
            self.grids.append(anchor_points)
            self.strids.append(strid_tensor)

        self.grids = torch.cat(self.grids, dim=0)
        self.strids = torch.cat(self.strids, dim=0)
        self.weight = torch.arange(
            self.reg_max, dtype=torch.float32, device=self.device
        )

    def decode(self, pred_tensor):
        reg_dist = pred_tensor.view(-1, self.num_bboxes, 4, self.reg_max)
        reg_prob = F.softmax(reg_dist, dim=-1)
        dist_projected = torch.sum(reg_prob * self.weight, dim=-1)
        dist_pixels = dist_projected * self.strids

        x1 = self.grids[..., 0] - dist_pixels[..., 0]
        y1 = self.grids[..., 1] - dist_pixels[..., 1]
        x2 = self.grids[..., 0] + dist_pixels[..., 2]
        y2 = self.grids[..., 1] + dist_pixels[..., 3]

        return torch.stack([x1, y1, x2, y2], dim=-1)



if __name__ == "__main__":
    conf = [(8, 48, 80), (16, 24, 40), (32, 12, 20)]
    decoder = BBoxDecoder(
        conf=conf, num_bboxes=5040, reg_max=16, device=torch.device("cuda")
    )

    pred_tensor = torch.randn(12, 5040, 64, device=torch.device("cuda"))
    bboxes = decoder.decode(pred_tensor)
