import torch.nn as nn

from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core import bbox2roi
from mmcv.cnn import normal_init


__all__ = ["RPN_Modulator", "RCNN_Modulator"]


class RPN_Modulator(nn.Module):
    def __init__(
        self,
        roi_out_size=7,
        roi_sample_num=2,
        channels=256,
        strides=[4, 8, 16, 32],
        featmap_num=5,
    ):
        super(RPN_Modulator, self).__init__()
        # TODO: 理解 SingleRoIExtractor 如何实现
        self.roi_extractor = SingleRoIExtractor(
            roi_layer={
                "type": "RoIAlign",
                "out_size": roi_out_size,
                "sample_num": roi_sample_num,
            },
            out_channels=channels,
            featmap_strides=strides,
        )
        self.proj_modulator = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, roi_out_size, padding=0)
                for _ in range(featmap_num)
            ]
        )
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1, padding=0) for _ in range(featmap_num)]
        )

    def forward(self, feats_z, feats_x, gt_bboxes_z):
        # TODO: rpn_modulator(z, x, gt_bboxes_z)
        # 目标检测中的 inference 是指的什么操作？
        return self.inference(feats_x, modulator=self.learn(feats_z, gt_bboxes_z))

    # TODO: 主要是在该函数中实现了 inference 的过程，也就是？
    # feature_x - feature_z <-- correlation 通过卷积实现
    # 实现 encode 的过程，encode 的结果为 out_ij，对应到公式中为 f_out(f_x(x) * f_z(z))
    # 这里 f_z(z) 被提前处理，处理之后的结果为 modulator(in learn(feature_z, gt_bboxes_z) function)
    # 这部分需要使用 transformer 进行 encode (将 query 的信息融合到 search image 也就是当前帧中)，是否需要进行 decode 后续考虑?
    def inference(self, feats_x, modulator):
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(modulator[i])
            for j in range(n_instances):
                query = modulator[i][j : j + 1]  # torch.Size([256, 7, 7])
                # TODO: 这里 f 代表什么
                gallary = [f[i : i + 1] for f in feats_x]
                out_ij = [
                    self.proj_modulator[k](query) * gallary[k]
                    for k in range(len(gallary))
                ]
                # TODO: 最后输出的 decoder 之后的 x-feature map，同时为特定的 query rpn 编码后的特征图
                # guide the detector to locate query-specific instances
                out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                yield out_ij, i, j

    def learn(self, feats_z, gt_bboxes_z):
        # 将目标从 Query Image 裁剪出来
        rois = bbox2roi(gt_bboxes_z)
        # 这一步应该就是 RoI Align 将 GT 从 feature map 中裁剪出来
        bbox_feats = self.roi_extractor(feats_z[: self.roi_extractor.num_inputs], rois)
        # 这里要弄清楚到底是不是在进行编码，还是说在进行一个 projection 的操作
        # 解答：这里不是编码过程，只是在对 query image 的特征进行一个处理转换
        # 将经过 RoI Align 得到的 feature map 和 Search Region 对应起来，对应到其中的位置？
        # modulation 调制，调控，调节
        modulator = [bbox_feats[rois[:, 0] == j] for j in range(len(gt_bboxes_z))]
        return modulator

    def init_weights(self):
        for m in self.proj_modulator:
            normal_init(m, std=0.01)
        for m in self.proj_out:
            normal_init(m, std=0.01)


class RCNN_Modulator(nn.Module):
    def __init__(self, channels=256):
        super(RCNN_Modulator, self).__init__()
        self.proj_z = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_x = nn.Conv2d(channels, channels, 3, padding=1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, z, x):
        return self.inference(x, self.learn(z))

    def inference(self, x, modulator):
        # assume one image and one instance only
        assert len(modulator) == 1
        return self.proj_out(self.proj_x(x) * modulator)

    def learn(self, z):
        # assume one image and one instance only
        assert len(z) == 1
        return self.proj_z(z)

    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)
