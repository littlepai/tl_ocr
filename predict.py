
import torch
from models import crnn
import params
import os
import PIL
from PIL import Image
import numpy as np
from dataset import converter
from utils import device
from PIL.PngImagePlugin import PngImageFile
from tqdm import tqdm
from io import BytesIO
import shutil
import requests



model = crnn.CRNN(params.image_channel, params.num_classes, params.nh)
model = model.to(device)
model_path = os.path.join(params.experiment, "crnn_best.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("从 {} 恢复模型参数".format(model_path))
else:
    print("\n未找到模型参数，请确认存在模型参数文件 {}，是否文件名称不对\n".format(model_path))
    exit(1)

# 传入一个 PIL.Image打开 并做 .convert("L") 处理的图片实例
def infer(img, model=model):
    img_array = np.array(img).astype(np.float32)/255.
    img_array = np.reshape(img_array, [1, params.image_channel, params.image_height, params.image_width])

    img_array = torch.from_numpy(img_array)
    img_array = img_array.to(device)

    preds = model(img_array)
    batch_size = preds.size(1)
    max_seq = torch.IntTensor([preds.size(0)] * batch_size) # T, B, C
    max_seq = max_seq.to(device)
    _, paths = preds.max(2)
    paths = paths.transpose(1, 0).contiguous().view(-1)
    label = converter.decode(paths.data, max_seq.data, raw=False)
    return label

# 接受图片验证码的路径或者是 PIL.Image 打开的图片实例（支持传入图片的列表）
def predict(imgorpath, model=model):
    if isinstance(imgorpath, str):
        img = Image.open(imgorpath).convert("L")
        return infer(img, model=model)
    elif isinstance(imgorpath, PngImageFile):
        img = imgorpath.convert("L")
        return infer(img, model=model)
    
    if isinstance(imgorpath, list):
        pred_labels = []
        for eachimg in imgorpath:
            pred_labels.append(predict(eachimg, model=model))
        return pred_labels
    else:
        print("请传入图片地址(或者图片列表) 或者 PIL.Image 打开的图片实例.")
        return None

# 下载知乎的线上真实的验证码，并验证
def predict_online(num=50, dirname="mark"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.zhihu.com",
        "Upgrade-Insecure-Requests": "1",}
    session = requests.Session()
    session.headers = headers #为了伪装，设置headers
    
    if os.path.exists(dirname):
        shutil.rmtree('mark')
    os.mkdir(dirname)
    captchaURL = r"https://www.zhihu.com/captcha.gif?type=login"
    print("\n\n"+"*"*50)
    print("开始从线上下载并测试验证码识别功能")
    for i in tqdm(range(num), ncols=50):
        img = Image.open(BytesIO(session.get(captchaURL).content))
        img = img.convert("L")
        expression = infer(img, model=model)
        img.save(os.path.join(dirname, expression+".png"))
    print("\n在当前目录下有mark文件夹，里面是刚刚识别的结果，如果有兴趣，可以统计正确率\n")

if __name__ == "__main__":
    predict_online()