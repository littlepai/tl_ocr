train_sample_num=200000 # 训练样本数
train_sample_dir=data/train

val_sample_num=20000 # 验证样本数
val_sample_dir=data/val



echo "开始生成训练数据 $train_sample_num 条，存于 $train_sample_dir"
python data_gen.py $train_sample_num  $train_sample_dir

echo "开始生成验证数据 $val_sample_num 条，存于 $val_sample_dir"
python data_gen.py $val_sample_num $val_sample_num

python main.py --epoch 10

