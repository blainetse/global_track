**从一开始方向就错了！只需要知道每一个作者的模型的模块的输入输出即可，不必探究 `mmdetection` 源码。
主要知道运行的原理即可，怎么调用的一个过程，怎么使用的。**

总结：只需要查看作者写的文件即可，`datasets` 文件另外，本来是可以直接使用 `DATASETS.register_module` 进行注册使用的，但是作者没有这样做，因此只需要在用到的时候查看该部分的代码即可。

1. Start from: `test_global_track.py`
  - 配置文件的加载
  - 预训练模型的加载
  - 数据相关处理与加载：`_submodules/neuron/neuron/data/transforms/pair_transforms/mmdet_transforms.py`
  这里主要使用 `BasicPairTransforms` 类来实现。

2. 




## two stage 目标检测相关知识

基本结构：`backbone --> neck --> head`

- https://zhuanlan.zhihu.com/p/342011052#:~:text=Neck%E6%98%AF%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E6%A1%86%E6%9E%B6%E4%B8%AD%E6%89%BF%E4%B8%8A%E5%90%AF%E4%B8%8B%E7%9A%84%E5%85%B3%E9%94%AE%E7%8E%AF%E8%8A%82%E3%80%82%20%E5%AE%83%E5%AF%B9Backbone%E6%8F%90%E5%8F%96%E5%88%B0%E7%9A%84%E9%87%8D%E8%A6%81%E7%89%B9%E5%BE%81%EF%BC%8C%E8%BF%9B%E8%A1%8C%E5%86%8D%E5%8A%A0%E5%B7%A5%E5%8F%8A%E5%90%88%E7%90%86%E5%88%A9%E7%94%A8%EF%BC%8C%E6%9C%89%E5%88%A9%E4%BA%8E%E4%B8%8B%E4%B8%80%E6%AD%A5head%E7%9A%84%E5%85%B7%E4%BD%93%E4%BB%BB%E5%8A%A1%E5%AD%A6%E4%B9%A0%EF%BC%8C%E5%A6%82%E5%88%86%E7%B1%BB%E3%80%81%E5%9B%9E%E5%BD%92%E3%80%81keypoint%E3%80%81instance,mask%E7%AD%89%E5%B8%B8%E8%A7%81%E7%9A%84%E4%BB%BB%E5%8A%A1%E3%80%82%20%E6%9C%AC%E6%96%87%E5%B0%86%E5%AF%B9%E4%B8%BB%E6%B5%81Neck%E8%BF%9B%E8%A1%8C%E9%98%B6%E6%AE%B5%E6%80%A7%E6%80%BB%E7%BB%93%E3%80%82
- https://blog.csdn.net/baidu_30594023/article/details/82623623
- https://www.jianshu.com/p/014e76d3b614
