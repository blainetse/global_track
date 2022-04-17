import os
import os.path as osp
import csv
import numpy as np

import neuron.ops as ops
from trackers.global_track import GlobalTrack
from .evaluator import Evaluator
from neuron.config import registry
from neuron.data import datasets
from neuron.models.trackers import OxUvA_Tracker


__all__ = ["OxUvA_Eval", "EvaluatorOxUvA"]


@registry.register_module
class OxUvA_Eval(Evaluator):
    r"""Evaluation pipeline and evaluation toolkit for OxUvA dataset.

    Args:
        dataset (Dataset): An OxUvA-like dataset.
    """

    def __init__(
        self,
        dataset,
        result_dir="results",
        report_dir="reports",
        visualize=False,
        plot_curves=False,
    ):
        self.dataset = dataset
        self.result_dir = osp.join(result_dir, self.dataset.name)
        self.report_dir = osp.join(report_dir, self.dataset.name)
        self.visualize = visualize
        self.plot_curves = plot_curves

    def run(self, tracker, visualize=None):
        if visualize is None:
            visualize = self.visualize
        # sanity check
        if not isinstance(tracker, OxUvA_Tracker):
            raise ValueError("Only supports trackers that implement OxUvA_Tracker.")
        ops.sys_print("Running tracker %s on %s..." % (tracker.name, self.dataset.name))

        # loop over the complete dataset
        # enumerate只是将里面的数据整体打包，并为这一个整体创建一个从 0 开始的索引
        for s, (img_files, target) in enumerate(self.dataset):
            """
            - `s`: 视频图像帧索引
            - `img_files`: 测试图像帧
            - `target`: 目标真实的信息，包括
                - `annotation`: 注解，目标真实的位置坐标框 (1, 4)
                - `meta`: 微调的 boundingbox 参数 (47, ) 表示抽取的 47 个视频帧
            """
            seq_name = self.dataset.seq_names[s]
            ops.sys_print("--Sequence %d/%d: %s" % (s + 1, len(self.dataset), seq_name))

            # TODO: skip if results exist 如果保存追踪测试结果的文件存在，就直接跳过
            record_file = osp.join(self.result_dir, tracker.name, f"{self.dataset.name.split('_')[-1]}/csvfile", "%s.csv" % seq_name)
            if osp.exists(record_file) and os.path.getsize(record_file):
                ops.sys_print("  Found results, skipping %s" % seq_name)
                continue

            # TODO: tracking loop pred怎么得到的，数据格式如何？
            preds, times = tracker.forward_test(
                img_files, target["anno"][0, :], visualize=visualize
            )
            assert len(preds) == len(img_files)

            # record results
            self._record(record_file, preds, times, seq_name, target["meta"])

    def report(self, tracker_names, plot_curves=None):
        raise NotImplementedError(
            "Evaluation of OxUvA results is not implemented."
            "Please submit the results to http://oxuva.net/ for evaluation."
        )

    def _record(self, record_file, preds, times, seq_name, meta):
        fields = [
            "video",
            "object",
            "frame_num",
            "present",
            "score",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
        ]
        vid_id, obj_id = seq_name.split("_")
        img_width, img_height = meta["width"], meta["height"]

        # record predictions
        record_dir = osp.dirname(record_file)
        if not osp.isdir(record_dir):
            os.makedirs(record_dir)
        with open(record_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            # TODO: 把追踪的结果写入到文件中
            # pred 代表追踪的目标信息
            for t, pred in preds.items():
                row = {
                    "video": vid_id,
                    "object": obj_id,
                    "frame_num": t,
                    "present": str(pred["present"]).lower(),
                    "score": pred["score"],
                    "xmin": pred["xmin"] / img_width,
                    "xmax": pred["xmax"] / img_width,
                    "ymin": pred["ymin"] / img_height,
                    "ymax": pred["ymax"] / img_height,
                }
                writer.writerow(row)
        ops.sys_print("  Results recorded at %s" % record_file)

        # record running times
        time_dir = osp.join(record_dir, "../times")
        if not osp.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = osp.join(
            time_dir, osp.basename(record_file).replace(".csv", "_time.txt")
        )
        np.savetxt(time_file, times, fmt="%.8f")


class EvaluatorOxUvA(OxUvA_Eval):
    r"""Evaluation pipeline and evaluation toolkit for OxUvA dataset.

    Args:
        root_dir (string): Root directory of OxUvA dataset.
        subset (string, optional): Specify ``dev`` or ``test``
            subset of OxUvA.
    """

    def __init__(self, root_dir=None, subset="dev", frame_stride=30, **kwargs):
        dataset = datasets.OxUvA(root_dir, subset=subset, frame_stride=frame_stride)
        super(EvaluatorOxUvA, self).__init__(dataset, **kwargs)
