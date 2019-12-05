# Pytorch 版验证码识别（包括迁移学习知乎的验证码）
[TensorFlow版本](https://github.com/littlepai/Unofficial-Zhihu-API)

## 获取代码
```
git clone git@github.com:littlepai/tl_ocr.git
```
## 安装依赖
```
pip install -r requirement.txt
```

## 思路步骤
1. 用软件生成接近知乎验证码的图片，20万到50万（下面简称 仿真验证码）
2. 用 仿真验证码 进行训练
3. 解压少量真实知乎验证码，用之前训练的模型参数，继续训练真实知乎验证码
4. 测试

**注意**
1. 由于Pytorch没有提供高效的解码方式，自己写的或者其他作者写的 Beam search 解码起来太慢了，所以案例使用贪心算法做解码，**不过没关系**，因为训练和解码是两条独立的线，训练照样用CTC Loss当损失函数，哪天不想用贪心算法做解码了，可以不用重新训练，直接把解码算法换成别的，都是可以的，记住，这两条线是独立的哦

2. 如果用GPU训练，显存最好大于等于6G，如果你的显卡实在low，导致溢出，那就修改一下run_tl.sh和run.sh 最后一行的启动训练命令，把每一批样本数改小点，加上参数 ** --val_batchSize 32 --batchSize 32**

### 操作步骤
```
sh run.sh       # 时间长短得看有无GPU，性能如何
sh run_tl.sh    # 第一个训练结束之后，这个是迁移学习，比较快
```

### 测试
```
python predict.py
```
