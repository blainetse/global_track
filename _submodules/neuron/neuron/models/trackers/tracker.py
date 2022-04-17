import numpy as np
import time

import neuron.ops as ops
from neuron.models.model import Model


__all__ = ["Tracker", "OxUvA_Tracker"]


class Tracker(Model):
    def __init__(
        self, name, is_deterministic=True, input_type="image", color_fmt="RGB"
    ):
        assert input_type in ["image", "file"]
        assert color_fmt in ["RGB", "BGR", "GRAY"]
        super(Tracker, self).__init__()
        self.name = name
        self.is_deterministic = is_deterministic
        self.input_type = input_type
        self.color_fmt = color_fmt

    def init(self, img, init_bbox):
        raise NotImplementedError

    def update(self, img):
        raise NotImplementedError

    def forward_test(self, img_files, init_bbox, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        bboxes[0] = init_bbox
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            if self.input_type == "image":
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == "file":
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bboxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])

        return bboxes, times


class OxUvA_Tracker(Tracker):
    def update(self, img):
        r"""One needs to return (bbox, score, present) in
        function `update`.
        """
        raise NotImplementedError

    # def update(self, imfile):
    #     import cv2
    #     import oxuva
    #     im = cv2.imread(imfile, cv2.IMREAD_COLOR)
    #     imheight, imwidth, _ = im.shape
    #     ok, cvrect = self._tracker.update(im)
    #     if not ok:
    #         return oxuva.make_prediction(present=False, score=0.0)
    #     else:
    #         rect = rect_from_opencv(cvrect, imsize_hw=(imheight, imwidth))
    #         return oxuva.make_prediction(present=True, score=1.0, **rect)

    def forward_test(self, img_files, init_bbox, visualize=False):
        # state variables
        frame_num = len(img_files)
        bboxes = np.zeros((frame_num, 4))
        bboxes[0] = init_bbox
        times = np.zeros(frame_num)

        preds = [
            {
                "present": True,
                "score": 1.0,
                "xmin": init_bbox[0],
                "xmax": init_bbox[2],
                "ymin": init_bbox[1],
                "ymax": init_bbox[3],
            }
        ]

        for f, img_file in enumerate(img_files):
            if self.input_type == "image":
                img = ops.read_image(img_file, self.color_fmt)
            elif self.input_type == "file":
                img = img_file

            begin = time.time()
            if f == 0:
                self.init(img, init_bbox)
            else:
                bbox, score, present = self.update(img)
                preds.append(
                    {
                        "present": present,
                        "score": score,
                        "xmin": bbox[0],
                        "xmax": bbox[2],
                        "ymin": bbox[1],
                        "ymax": bbox[3],
                    }
                )
                bboxes[f, :] = bbox
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, bboxes[f, :])

        # update the preds as one-per-second
        frame_stride = 30
        preds = {f * frame_stride: pred for f, pred in enumerate(preds)}

        return preds, times


def rect_to_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs = rect["xmin"] * imwidth
    ymin_abs = rect["ymin"] * imheight
    xmax_abs = rect["xmax"] * imwidth
    ymax_abs = rect["ymax"] * imheight
    return (xmin_abs, ymin_abs, xmax_abs - xmin_abs, ymax_abs - ymin_abs)


def rect_from_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs, ymin_abs, width_abs, height_abs = rect
    xmax_abs = xmin_abs + width_abs
    ymax_abs = ymin_abs + height_abs
    return {
        "xmin": xmin_abs / imwidth,
        "ymin": ymin_abs / imheight,
        "xmax": xmax_abs / imwidth,
        "ymax": ymax_abs / imheight,
    }
