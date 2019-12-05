import models.crnn as crnn
import torch
from torch import nn
import torch.optim as optim
import params
from torch.utils.data import Dataset, DataLoader
from dataset import batchify, OcrDataset
import utils
from utils import device
from dataset import converter
# from beam_search import beam_decode


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(model, train_loader, criterion, iteration, optimizer):

    # for p in model.parameters():
    #     p.requires_grad = True
    model.train()
    loss_avg = utils.averager()
    for i_batch, (image, label, length) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        length = length.to(device)
        
        preds = model(image)
        batch_size = preds.size(1)
        max_seq = torch.IntTensor([preds.size(0)] * batch_size) # T, B, C
        max_seq = max_seq.to(device)
        # print(preds.shape, text.shape, max_seq.shape, length.shape)
        # torch.Size([41, 16, 6736]) torch.Size([160]) torch.Size([16]) torch.Size([16])
        cost = criterion(preds, label, max_seq, length) / batch_size # 一个样本的 cost
        model.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch+1) % params.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (iteration, params.epoch, i_batch, len(train_loader), loss_avg.val()))
            loss_avg.reset()

def val(model, val_loader, criterion, iteration, max_i=1000):

    print('Start val')
    # for p in model.parameters():
    #     p.requires_grad = False
    model.eval()
    i = 0
    n_correct = 0
    total_images_count = 0
    
    loss_avg = utils.averager()
    
    for i_batch, (image, label, length) in enumerate(val_loader):
        image = image.to(device)
        label = label.to(device)
        length = length.to(device)
        
        preds = model(image) # preds 所在设备跟会跟model相同
        batch_size = preds.size(1)
        max_seq = torch.IntTensor([preds.size(0)] * batch_size) # T, B, C
        max_seq = max_seq.to(device)
        cost = criterion(preds, label, max_seq, length) / batch_size # 一个样本的 cost
        loss_avg.add(cost)
        _, paths = preds.max(2)
        # paths, scores = beam_decode(paths)
        paths = paths.transpose(1, 0).contiguous().view(-1)
        pred_labels = converter.decode(paths.data, max_seq.data, raw=False)
        label_split_start_idx = 0
        for pred, target_len in zip(pred_labels, length):
            target = label[label_split_start_idx:label_split_start_idx+target_len]
            label_split_start_idx += target_len.tolist() # 这个 tolist 惊到我了，其实是一个 int
            target = "".join([converter.alphabet[t-1] for t in target.cpu().numpy()])
            # target = converter.decode(target, torch.IntTensor([len(target)]))
            if pred == target:
                n_correct += 1
            total_images_count += 1

        if (i_batch+1)%params.displayInterval == 0:
            print('[%d/%d][%d/%d]' %
                      (iteration, params.epoch, i_batch, len(val_loader)))

        if i_batch == max_i:
            break
    
    # 验证样本展示，方便观察训练情况
    raw_preds = converter.decode(paths.data, max_seq.data, raw=True)[:params.n_test_disp]
    label_split_start_idx = 0
    for raw_pred, pred, target_len in zip(raw_preds, pred_labels, length):
        target = label[label_split_start_idx:label_split_start_idx+target_len]
        label_split_start_idx += target_len.tolist() # 这个 tolist 惊到我了，其实是一个 int
        target = "".join([converter.alphabet[t-1] for t in target.cpu().numpy()])
        # target = converter.decode(target, torch.IntTensor([len(target)])) # 这个 decode 是解码路径，不能做label的int2char,因为比如label=1111,那decode会变成a,注意别混淆了
        print('%-20s => %-20s, tg: %-20s' % (raw_pred, pred, target))

    # print(n_correct)
    # print(max_i * params.val_batchSize)
    accuracy = n_correct / float(total_images_count)
    print('Val loss: %f, accuray: %d/%d=%f' % (loss_avg.val(), n_correct, total_images_count, accuracy))

    return accuracy

def main(model, train_loader, val_loader, criterion, optimizer):
    model = model.to(device)
    criterion = criterion.to(device)
    Iteration = 0
    best_accuracy = None # params.minimum_accuracy
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',verbose=True,patience=3)
    while Iteration < params.epoch:
        # print(optimizer)
        # adjust_learning_rate(optimizer,epoch) # 根据迭代轮次来调整学习率
        train(model, train_loader, criterion, Iteration, optimizer) # 训练
        # max_i: cut down the consuming time of testing, if you'd like to validate on the whole testset, please set it to len(val_loader)
        accuracy = val(model, val_loader, criterion, Iteration, max_i=1000) # 验证
        scheduler.step(accuracy) # 按照一定策略更新学习率
        # for p in model.parameters():
        #     p.requires_grad = True
        # 比较准确率，保存最好的模型
        if best_accuracy is None:
            best_accuracy = accuracy
        elif accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.experiment, Iteration, accuracy))
            torch.save(model.state_dict(), '{0}/{1}'.format(params.experiment, params.state_dict_name))
        print("is best accuracy: {0}, at epoch {1}".format(accuracy == best_accuracy, Iteration))
        Iteration+=1

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

if __name__ == "__main__":
    criterion = torch.nn.CTCLoss(reduction='sum')

    # 模型
    model = crnn.CRNN(params.image_channel, params.num_classes, params.nh)
    if params.restore:
        model_state_dict_path = "{0}/{1}".format(params.experiment, params.state_dict_name)
        print("从 {} 恢复参数".format(model_state_dict_path))
        model.load_state_dict(torch.load('{}'.format(model_state_dict_path), map_location=device))
    else:
        model.apply(weights_init)

    # 优化器
    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(model.parameters(), lr=params.lr,
                               betas=(params.beta1, params.beta2))
    elif params.adadelta:
        optimizer = optim.Adadelta(model.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=params.lr)

    # 数据迭代器
    print("================== 正在从 {} 加载 train 数据 ==================".format(params.traindir))
    train_data = OcrDataset(params.traindir)
    print("加载 train 数据 {} 条".format(len(train_data)))
    print("================== 正在从 {} 加载 val 数据 ==================".format(params.valdir))
    val_data = OcrDataset(params.valdir)
    print("加载 val 数据 {} 条".format(len(val_data)))

    train_loader = DataLoader(train_data, batch_size=params.batchSize, num_workers=params.workers, shuffle=True, collate_fn=batchify)
    val_loader = DataLoader(val_data, batch_size=params.val_batchSize, num_workers=params.workers, shuffle=True, collate_fn=batchify)
    
    model.register_backward_hook(backward_hook)
    # 训练
    main(model, train_loader, val_loader, criterion, optimizer)