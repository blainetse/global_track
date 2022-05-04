# USE CONDA

```shell
conda env create -f environment.yml
```
> 不推荐：安装过程中没有给用户确定选择的空间，直接默认安装，如果没有找到相匹配的版本，可能会自动安装 `CPU` 版本的 `PyTorch`。


# USE PIP

将所有的依赖包放入到 `requirements.txt` 文件中：
```yaml
black
imageio
mmcv==0.4.3
numpy
opencv-python
opencv-python-headless
Pillow
scikit-image
scikit-learn
scipy
Shapely
# 以下是使用 pip install 安装的配置
torch==1.8.1+cu111
torchaudio==0.8.1
torchvision==0.9.1+cu111
-f https://download.pytorch.org/whl/torch_stable.html
```
然后使用 `pip install -r requirements.txt` 即可，该过程有可选性，推荐使用。

# 总结

1. 使用 `conda create -n name python=3.8`
2. `pip install -r requirements.txt`

按照如上的方法具有更多的可选性。


# 运行问题及解决

1. `AT_CHECK --> TORCH_CHECK`
2. 修改 `from ... import ...`


在 `train` 部分，如果数据集出错，但是已经创建好了 `cache/*.pkl` 文件，在修改对应的源代码之后，重新调试运行之前，记得将之前的缓存文件删除，否则还是会报之前一样的错误！
