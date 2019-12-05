
import os
import params
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
from utils import strLabelConverter

converter = strLabelConverter(params.charset)

# from params import charset
# # import string
# # charset = string.digits + string.ascii_lowercase#'0123456789+-*()'
# encode_maps = {}
# decode_maps = {}
# for i, char in enumerate(charset, 1):
#     encode_maps[char] = i
#     decode_maps[i] = char

# SPACE_INDEX = 0
# SPACE_TOKEN = ''
# encode_maps[SPACE_TOKEN] = SPACE_INDEX
# decode_maps[SPACE_INDEX] = SPACE_TOKEN

def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, 
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    xs = []
    ys = []
    lens = []
    for x,y,l in data:
        xs.append(x)
        ys.extend(y)
        lens.append(len(y))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long), torch.tensor(lens, dtype=torch.long)
    

class OcrDataset:
    def __init__(self, data_dir):
        self.image = [] # 所有的图片都加载进内存待等待提取，加载进内存当然是为了速度了
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                # im = np.array(Image.open(image_name)).astype(np.float32)/255.
                im = np.array(Image.open(image_name).convert("L")).astype(np.float32)/255.
                # im = np.array(Image.open(image_name).convert("L").point(lambda x: 0 if x < 150 else 1)).astype(np.float32)
                # im = cv2.imread(image_name, 0).astype(np.float32)/255.
                # resize to same height, different width will consume time on padding
                # im = cv2.resize(im, (image_width, image_height))
                im = np.reshape(im, [params.image_channel, params.image_height, params.image_width])

                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split(os.sep)[-1].split('_')[1].split('.')[0] # code 是验证码
                # code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)] # code转成[1,2,3,4] 字码列表
                code, _ = converter.encode(code)
                self.labels.append(code.tolist())
    
    # 使size方法变成属性，调用的时候self.size即可，不用调用self.size() #这里体现不出@property的优点
    @property
    def size(self):
        return len(self.labels)
    
    def __len__(self):
        return len(self.labels)

    # 给定index, 抽取labels
    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels
    
    def __getitem__(self, index):
        return self.image[index], self.labels[index], [78]*len(self.labels[index])
    # # 给定index, 得到一个批次的训练数据
    # def input_index_generate_batch(self, index=None):
    #     if index:
    #         image_batch = [self.image[i] for i in index]
    #         label_batch = [self.labels[i] for i in index]
    #     else:
    #         image_batch = self.image
    #         label_batch = self.labels

    #     def get_input_lens(sequences):
    #         # 分片的序列长度，因为验证码图片序列长度都是一样的，不像句子有长有短
    #         # 所以这里的长度都是一样的
    #         lengths = np.asarray([params.max_stepsize for _ in sequences], dtype=np.int64)

    #         return sequences, lengths

    #     batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
    #     batch_labels = sparse_tuple_from_label(label_batch) # 转成稀疏矩阵

    #     return batch_inputs, batch_seq_len, batch_labels

# class OcrDataset2(Dataset):
#     def __init__(self, img_root, label_path,transforms=None):
#         super(OcrDataset, self).__init__()
#         self.img_root = img_root
#         # self.isBaidu = isBaidu
#         self.labels = self.get_labels(label_path)
#         # print(self.labels[:10])
#         # self.alphabet = alphabet
#         self.transforms = transforms
#         self.width, self.height = resize
#         # print(list(self.labels[1].values())[0])
#     def get_labels(self, label_path):
#         # return text labels in a list
#         if self.isBaidu:
#             with open(label_path, 'r', encoding='utf-8') as file:
#                 # {"image_name":"chinese_text"}
#                 content = [[{c.split('\t')[2]:c.split('\t')[3][:-1]},{"w":c.split('\t')[0]}] for c in file.readlines()];
#             labels = [c[0] for c in content]
#             # self.max_len = max([int(list(c[1].values())[0]) for c in content])
#         else:
#             with open(label_path, 'r', encoding='utf-8') as file:
#                 labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]    
#         return labels


#     def __len__(self):
#         return len(self.labels)

#     # def compensation(self, image):
#     #     h, w = image.shape # (48,260)
#     #     image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
#     #     # if w>=self.max_len:
#     #     #     image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
#     #     # else:
#     #     #     npi = -1*np.ones(self.max_len-)

#     #     return image
#     def preprocessing(self, image):

#         ## already have been computed
#         image = image.astype(np.float32) / 255.
#         image = torch.from_numpy(image).type(torch.FloatTensor)
#         image.sub_(params.mean).div_(params.std)

#         return image

#     def __getitem__(self, index):
#         image_name = list(self.labels[index].keys())[0]
#         # label = list(self.labels[index].values())[0]
#         image = cv2.imread(self.img_root+'/'+image_name)
#         # print(self.img_root+'/'+image_name)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         h, w = image.shape
#         # Data augmentation
#         # width > len ==> resize width to len
#         # width < len ==> padding width to len 
#         # if self.isBaidu:
#         #     # image = self.compensation(image)
#         #     image = cv2.resize(image, (0,0), fx=160/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
#         image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
#         image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
#         image = self.preprocessing(image)

#         return image, index

