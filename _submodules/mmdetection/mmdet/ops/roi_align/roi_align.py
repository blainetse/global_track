import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import roi_align_cuda


class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(
                features, rois, out_h, out_w, spatial_scale, sample_num, output
            )
        else:
            raise NotImplementedError

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None and grad_output.is_cuda

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(
                batch_size, num_channels, data_height, data_width
            )
            roi_align_cuda.backward(
                grad_output.contiguous(),
                rois,
                out_h,
                out_w,
                spatial_scale,
                sample_num,
                grad_input,
            )

        return grad_input, grad_rois, None, None, None


roi_align = RoIAlignFunction.apply


# TODO: 理解该段如何实现
class RoIAlign(nn.Module):
    r"""个人理解：结合网络结构图，该类的主要功能是将目标从 feature map 中裁剪出来
    注意：目标检测中经过 backbone 得到的 feature map 和目标追踪的不同。
        前者得到的不一定是正方形，而没有使用 twostage 即使用目标检测的孪生网络的目标追踪算法得到的为正方形。
        使用目标检测的进行追踪的好处是：没有许多先验假设在里面，不考虑物体是如何移动的等等，
        但是该方法的缺点是：容易受背景信息的干扰（个人理解），运动模糊对该方法的影响也挺大（从目标检测的角度来看）
    """
    def __init__(self, out_size, spatial_scale, sample_num=0, use_torchvision=False):
        super(RoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align

            return tv_roi_align(
                features, rois, self.out_size, self.spatial_scale, self.sample_num
            )
        else:
            return roi_align(
                features, rois, self.out_size, self.spatial_scale, self.sample_num
            )

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += "(out_size={}, spatial_scale={}, sample_num={}".format(
            self.out_size, self.spatial_scale, self.sample_num
        )
        format_str += ", use_torchvision={})".format(self.use_torchvision)
        return format_str
