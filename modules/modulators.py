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
        # [ ] 功能：给一个权重，用于训练？
        self.proj_modulator = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, roi_out_size, padding=0)
                for _ in range(featmap_num)
            ]
        )
        # [ ] encoder?
        self.proj_out = nn.ModuleList(
            # 将 RoI Align 之后的图片转换为 1*1 的大小进行编码
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
        # TODO: 搞清楚这里的 feature_x.size() ?= 以及最后一个维度为什么不等：使用 FPN 的原因
        # feats_x[0] = [1, 256, 192, 336]  ==> ×16
        # feats_x[1] = [1, 256, 96, 168]  ==> ×8
        # feats_x[2] = [1, 256, 48, 84]  ==> ×4
        # feats_x[3] = [1, 256, 24, 42]  ==> ×2
        # feats_x[3] = [1, 256, 12, 21]  ==> ×1
        # 这里为什么只取最后 ×16 倍的结果？底层包含了丰富的位置信息，顶层包含语义信息
        # 这里我们只需要 bbox，因此只取底层的即可
        n_imgs = len(feats_x[0])
        for i in range(n_imgs):
            n_instances = len(modulator[i])
            for j in range(n_instances):
                # torch.Size([256, 7, 7]) 遍历 i-th 张图像帧的 RoI Align 后的 feature map
                query = modulator[i][j : j + 1]
                # TODO: 这里 f 代表什么：f 代表特征，取出 feats_x 列表中每一个 feature
                # featmap_num=5 总共 5 个特征图，每一个 feature map 的 shape = [1, 256, 192, 336]
                # 最后两个维度代表图像特征图的 size，经过 FPN 处理逐级减半
                gallary = [f[i : i + 1] for f in feats_x]
                out_ij = [
                    # [ ] FPN project modulator 实现的功能？一个 Conv2D 操作
                    # 总共有 5 个 Conv2D，每一个负责一个 feature map，给一个权重，方便训练 RPN Loss?
                    self.proj_modulator[k](query) * gallary[k]
                    for k in range(len(gallary))
                ]
                # [ ] 最后输出的 decoder 之后的 x-feature map，同时为特定的 query rpn 编码后的特征图
                # guide the detector to locate query-specific instances
                # [ ] 这里实现的才是 encoder 操作，转换为 1×1 的矩阵后，使用 Conv2D 进行编码
                out_ij = [p(o) for p, o in zip(self.proj_out, out_ij)]
                yield out_ij, i, j  # Generator, 通过 next 进行调用的时候实现，产生对应的输出

    def learn(self, feats_z, gt_bboxes_z):
        # 将目标从 Query Image 裁剪出来
        # gt_bboxes_z: [1, 4]
        # rois: [1, 5] 增加了一个维度 batch_ind
        rois = bbox2roi(gt_bboxes_z)
        # 这一步应该就是 RoI Align 将 GT 从 feature map 中裁剪出来
        # [ ] roi_extractor.num_inputs = len(self.featmap_strides) = 4
        # feats_z, [1, 5] FPN 结构
        # feats_z[: 4], [1, 4] 只取前四个
        bbox_feats = self.roi_extractor(feats_z[: self.roi_extractor.num_inputs], rois)  # [1, 256, 7, 7]
        # 这里要弄清楚到底是不是在进行编码，还是说在进行一个 projection 的操作
        # 解答：这里不是编码过程，只是在对 query image 的特征进行一个处理转换
        # 将经过 RoI Align 得到的 feature map 和 Search Region 对应起来，对应到其中的位置？
        # modulation 调制，调控，调节
        # modulator 每个 gt_box 对应的 bbox_feat
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
        # z: [1, 256, 7, 7]
        # x: [1000, 256, 7, 7]
        return self.inference(x, self.learn(z))

    def inference(self, x, modulator):
        # assume one image and one instance only
        assert len(modulator) == 1
        # [ ] x: [1000, 256, 7, 7]
        # modulator: [1, 256, 7, 7]
        # self.proj_x(x): [1000, 256, 7, 7]
        # self.proj_x(x) * modulator: [1000, 256, 7, 7]
        # [1000, 256, 7, 7]
        return self.proj_out(self.proj_x(x) * modulator)

    def learn(self, z):
        # assume one image and one instance only
        assert len(z) == 1
        return self.proj_z(z)

    def init_weights(self):
        normal_init(self.proj_z, std=0.01)
        normal_init(self.proj_x, std=0.01)
        normal_init(self.proj_out, std=0.01)
