tl_train_sample_dir=data/tl_train
tl_val_sample_dir=data/tl_val

rm -rf $tl_train_sample_dir
echo "解压迁移样本当训练集,存放于 $tl_train_sample_dir"
unzip data/迁移学习样本.zip -d $tl_train_sample_dir > /dev/null

rm -rf $tl_val_sample_dir
echo "解压一毛一样的迁移样本当验证集（数据少）,存放于 $tl_val_sample_dir"
unzip data/迁移学习样本.zip -d $tl_val_sample_dir > /dev/null

echo "开始迁移学习"
python main.py  --restore True --lr 1e-5 --traindir $tl_train_sample_dir --valdir $tl_val_sample_dir --batchSize 32