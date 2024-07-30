# SFT语言模型 Step by Step

## 数据准备

对于短对话类，我们需要准备一个jsonl文件。其中每行包括三个部分：input，output和instruction。如：
```json
{"input": "“input”:基于1951—2014年2400余个中国国家级地面气象站均一化相对湿度资料,采用薄盘样条法,进行空间内插,得到了中国地面均一化相对湿度月值0.5°×0.5°格点数据集(CR数据集).数据集的质量评估结果表明:冬季插值误差相对夏季偏高.冬、春、夏、秋季分析值与站点实测值的插值偏差空间区域特征不显著,平均偏差分别为0.002%/月、0.013%/月、0.008%/月和0.007%/月.冬、春、夏、秋季平均相对误差分别为0.431%/月、0.439%/月、0.286%/月和0.382%/月.分析值与站点实测值间的平均相关系数达0.89.整体来看,插值后的格点化相对湿度资料能够比较准确、细致地描述我国年平均相对湿度场的东南湿、西北干的主要空间特征.能够较好地展现长江以南地区、黄河以南长江以北地区、西北地区、天山南北麓、塔里木盆地等大地形的相对湿度变化特征.由于青藏高原台站稀少,格点数据集对该地区空气相对湿度特征的刻画是否合理很难给出定性、定量的判断.通过对CR数据集进行长时间序列气候变化趋势分析,表明60年来全国平均相对湿度呈减小趋势.,“task”：{\"领域\": \"气候学/气象学\", \"专家\": \"气候学家/气象学家\", \"任务\": [\"收集并整理过去63年的中国国家级地面气象站的均一化相对湿度资料\", \"应用薄盘样条法进行空间内插，生成0.5°×0.5°格点数据集\", \"对生成的数据集进行质量评估，包括插值误差、偏差和相关系数的计算\", \"分析不同季节的数据集，找出各季节的插值特征和平均偏差\", \"研究长时间序列气候变化趋势，确定全国平均相对湿度的变化趋势\"]}", "output": "{\"entity_types\": [\"时间范围\", \"数据源\", \"地理区域\", \"技术方法\", \"数据分析结果\", \"气候变化趋势\"]}", "instruction": "根据input中的领域和任务，协助用户识别input文本中存在的实体类型。 实体类型必须与用户任务相关。 避免使用诸如“其他”或“未知”的通用实体类型。 非常重要的是：不要生成冗余或重叠的实体类型。用JSON格式输出。"}
```

## 数据打散并且重复

由于RWKV6的SFT以State为主，我们尽可能让state能过拟合。用linux shuf命令操作。假设我们的数据文件名是data.jsonl，输出到train_data_dir目录我们可以用如下命令：

```bash
shuf data.jsonl >> train_data_dir/data.jsonl
...
shuf data.jsonl >> train_data_dir/data.jsonl
```
运行十次，这样train_data_dir/data.jsonl内容是原文件重复了十次。

## 数据预处理
为了不在训练中浪费CPU去做tokenization，我们提前对数据进行tokenization。并且为了让长度不同的数据能够在一次训练中训练，但又不浪费显存，我们将数据按照长度分组。

假设我们保存jsonl文件的目录是train_data_dir，输出保存目录为train_data_ds_dir,我们可以用如下命令：
```bash
python data/SftUtilities.py --input_dir train_data_dir --output_dir train_data_ds_dir
```


## 训练

我们上一步在train_data_ds_dir中生成了训练数据，首先我们观察train_data_ds_dir生成了多少组长度数据。list一下train_data_ds_dir，我们可以看到类似如下的输出：
```bash
ls train_data_ds_dir
train_data_ds_dir_dataset_lds_1024  train_data_ds_dir_dataset_lds_2048  train_data_ds_dir_dataset_lds_256  train_data_ds_dir_dataset_lds_512
```
意味着我们有四组数据，分别是长度为1024，2048，256和512的数据。

后面的训练代码中会体现出这些数据长度的使用。

我们通过环境变量来控制训练类型，是State，还是Lora，还是Pissa。

State训练：
```bash
WKV= RWKV_TRAIN_TYPE=states CUDA_VISIBLE_DEVICES=0 python peft_train/peft_train_sft.py --train_data  train_data_ds_dir  --model_file models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth --output_dir models/trained/sft/my_train_data_states --train_type state --max_epoch 1 --grad_cp 1 --strategy deepspeed_stage_2_offload --wandb my_train_wandb --num_nodes 1 --num_devices 1 --dropout 0.01 --train_lengths 256 512 1024 2048 --train_batch_sizes 12 6 3 1 --lr_init 1 --lr_final 0.01
```
接下来我们解释一下上面的参数：

| 参数名 | 参数值 | 说明 |
| ------ | ------ | ---- |
| WKV |  | 控制扩展类型，可以是fla 空表示默认CUDA
| RWKV_TRAIN_TYPE |  states| 控制训练类型，可以是states、lora或pissa |
| CUDA_VISIBLE_DEVICES | 0 | 控制使用的GPU设备编号，如果用4个显卡，则输入0,1,2,3 |
| --train_data | train_data_ds_dir | 训练数据的目录 |
| --model_file | models/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth | 基础模型文件的路径 |
| --output_dir | models/trained/sft/my_train_data_states | 输出模型的目录 |
| --train_type | state | 训练类型，可以是state、lora或pissa |
| --max_epoch | 1 | 最大训练轮数。我们数据已经重复了10次，只需要训练一个epoch即可 |
| --grad_cp | 1 | 梯度累积的步数 |
| --strategy | deepspeed_stage_2_offload | 训练策略 |
| --wandb | my_train_wandb | Wandb项目名称 |
| --num_nodes | 1 | 训练节点数 |
| --num_devices | 1 | 训练设备数，只能小于等于CUDA_VISIBLE_DEVICES的树木|
| --dropout | 0.01 | Dropout的概率 |
| --train_lengths | 256 512 1024 2048 | 训练数据的长度，和前面生成的数据集包含的长度子集 |
| --train_batch_sizes | 12 6 3 1 | 对应长度的batch size，训练代码会根据这个设置，调整在不同长度下的batch size。 |
| --lr_init | 1 | 初始学习率 |
| --lr_final | 0.01 | 最终学习率 |

## 训练结果
训练结果保存在output_dir下面，只保存state文件本身，不保存原始基础模型。