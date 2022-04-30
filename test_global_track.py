import _init_paths
from _submodules.neuron.neuron.ops.io import download
import neuron.data as data
from trackers import *

if __name__ == "__main__":
    cfg_file = "configs/qg_rcnn_r50_fpn.py"
    ckp_file = "checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth"
    transforms = data.BasicPairTransforms(train=False)
    tracker = GlobalTrack(cfg_file, ckp_file, transforms, name_suffix="qg_rcnn_r50_fpn")
    evaluators = [
        ## start
        # data.EvaluatorOxUvA(root_dir="/data/OxUvA/", subset="dev"),  # on dev data
        data.EvaluatorOxUvA(root_dir="/data/OxUvA/", subset="test"),  # on test data
        # data.EvaluatorVOT(root_dir='/data/VOT2020', version=2020)
        # data.EvaluatorTLP(root_dir="/data/TLP/TLP"),
        ## end
        # data.EvaluatorOTB(root_dir='/data/OTB100', version=100),
        # data.EvaluatorLaSOT(root_dir='/data/LaSOT/LaSOTBenchmark', rame_stride=10),
        # data.EvaluatorGOT10k(subset='test')
    ]
    for e in evaluators:
        e.run(tracker, visualize=False)
        e.report(tracker.name)
