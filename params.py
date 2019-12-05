import string
import argparse
import sys

parser = argparse.ArgumentParser(description="progrom description")
# parser.add_argument('--minimum_accuracy', type=float, default=0.5, help="准确率高于次值才进入最优模型比较和保存")
parser.add_argument('--n_test_disp', type=int, default=10, help="验证的时候挑多少条展示")
parser.add_argument('--displayInterval', type=int, default=500, help="训练过程，每一个迭代中，每n批打印一次训练情况")
parser.add_argument('--experiment', type=str, default="./models", help="最优模型保存目录")
parser.add_argument('--adam', type=bool, default=False, help="是否使用adam优化器")
parser.add_argument('--adadelta', type=bool, default=False, help="是否使用adadelta优化器")
parser.add_argument('--nh', type=int, default=256, help="RNN 隐藏层大小")
parser.add_argument('--image_width', type=int, default=150, help="图片宽度")
parser.add_argument('--image_height', type=int, default=60, help="图片高度")
parser.add_argument('--image_channel', type=int, default=1, help="图片通道")
parser.add_argument('--max_stepsize', type=int, default=76, help="")
parser.add_argument('--num_classes', type=int, default=37, help='10 + 26 + 1 多加一个空白（CTC的特殊用法，数据处理的时候空白idx默认是0，代表啥也没有，跟空格" "不一样，这里不处理空格的情况）这里不处理空格" ",比如 "a b   cd" = "abcd", 如果需要处理空格，请在 charset 里面加上 " "')
parser.add_argument('--beta1', type=float, default=0.5, help="优化器参数")
parser.add_argument('--beta2', type=float, default=0.999, help="优化器参数")
parser.add_argument('--lr', type=float, default=1e-4, help="初始学习率")
parser.add_argument('--epoch', type=int, default=20, help="迭代轮次")
parser.add_argument('--charset', type=str, default=string.digits + string.ascii_lowercase, help="10个数字 + 26个小写字母")
parser.add_argument('--batchSize', type=int, default=128, help="训练集每批样本数")
parser.add_argument('--val_batchSize', type=int, default=128, help="验证集每批样本数")
parser.add_argument('--workers', type=int, default=3, help="数据加载器进程数")
parser.add_argument('--traindir', type=str, default="data/train", help="训练集目录")
parser.add_argument('--valdir', type=str, default="data/val", help="验证集目录")
parser.add_argument('--restore', type=bool, default=False, help="是否加载之前训练保存下载的 best model state_dict")
parser.add_argument('--state_dict_name', type=str, default="crnn_best.pth", help="是否加载之前训练保存下载的 best model state_dict")



args = parser.parse_args()


__thismodule__ = sys.modules[__name__]

for __k__, __v__ in args.__dict__.items():
    setattr(__thismodule__, __k__, __v__)
