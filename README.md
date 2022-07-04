<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-07-04 17:31:00
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-04 17:31:00
-->
<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-07-03 20:27:59
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-04 17:04:43
-->
# 基于CNN+LSTM的流量分析识别系统

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/CNN-LSTM-Flow-Analysis)
[![](https://img.shields.io/github/license/orion-orion/CNN-LSTM-Flow-Analysis)](https://github.com/orion-orion/CNN-LSTM-Flow-Analysis/blob/master/LICENSE)
[![](https://img.shields.io/github/stars/orion-orion/CNN-LSTM-Flow-Analysis?style=social)](https://github.com/orion-orion/CNN-LSTM-Flow-Analysis)


### 关于本项目
本项目为2020年中国高校计算机大赛(C4)－网络技术挑战赛EP2决赛赛项，题目为构建一个在线流量识分析与识别系统，能够实时识别出网络上的正常业务流量、恶意软件流量和网络攻击流量，并对各种流量随时序的变化进行进行可视化展示，我们采用CNN+LSTM时空神经网络使用，其中CNN对流量空间特征进行提取，LSTM对流量时序特征进行提取，从而完成不同种类流量分类功能。我们将思博伦官方给出的流量pcap包解析为流量的URL进行训练, 最终在官方给出的测试流量包上达到 93.5% 的准确率。

### 环境依赖
运行以下命令安装环境依赖：
```
pip install -r requirements.txt
```

## 数据集
已经搜集好的训练数据以`csv`格式存储于`data`目录下. `csv`文件第一行的列名需要为`["label", "content"]` 或者`["content", "label"]`（我们这里采用前者）。其中`0`标签表示为业务流量，`1`标签表示为网络攻击流量，`2`标签表示为恶意软件流量。

## 模型

您可以自由选择包括C-LSTM时空神经网络在内的以下模型使用
- LSTM分类器（参见`rnn_classifier.py`）。
- 双向LSTM分类器（参见`rnn_classifier.py`）。
- CNN分类器（参见`cnn_classifier`）。参考: [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
- C-LSTM分类器（参见`clstm_classifier.py`）。 参考: [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630).

我本地训练好的模型参数`params.pkl`以及向量化预处理文件`vocab`已经保存在`params`目录下，大家可直接加载使用。

## 项目目录说明

-data ------------------- 存放数据

-model ------------------- 存放模型参数

-params ------------------- 存放模型超参数相关信息和向量化预处理信息

-prediction ------------------- 存放对测试流量的预测结果

-summaries ------------------- 模型训练和验证信息（用于`tensorboard`可视化展示）

-clstm_classifier.py ------------------- C-LSTM分类器的实现

-cnn_classifier ------------------- CNN分类器的实现

-data_helper  ------------------- 完成预处理向量化操作

-rnn_classifier.py ------------------- LSTM分类器和双向LSTM分类器的实现

-test.py ------------------- 完成模型在测试集上的测试操作

-train.py ------------------- 完成模型在训练集上的训练操作和在验证集上的验证操作

## 使用方法
### 训练

运行`train.py`来训练模型。

参数如下:
```
optional arguments:
  --clf CLF             Type of classifiers. Default: clstm. You have four
                        choices: [cnn, lstm, blstm, clstm]
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Max document length
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Cross validation test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. For CNN, C-LSTM.
  --filter_sizes FILTER_SIZES
                        CNN filter sizes. For CNN, C-LSTM.
  --num_filters NUM_FILTERS
                        Number of filters per filter size. For CNN, C-LSTM.
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell. For LSTM, Bi-
                        LSTM
  --num_layers NUM_LAYERS
                        Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM
  --keep_prob KEEP_PROB
                        Dropout keep probability
  --learning_rate LEARNING_RATE
                        Learning rate
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --decay_rate DECAY_RATE
                        Learning rate decay rate. Range: (0, 1]
  --decay_steps DECAY_STEPS
                        Learning rate decay steps.
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate the model on validation set after this many
                        steps
  --save_every_steps SAVE_EVERY_STEPS
                        Save the model after this many steps
  --num_checkpoint NUM_CHECKPOINT
                        Number of models to store
```
您可以运行`train.py`开始训练。例如:
```
python train.py --data_file=data/data.csv --clf=clstm
```

训练完成后, 你可以使用`tensorboard`来查看计算图, 损失函数和评价指标的可视化:  

```
tensorboard --logdir=summaries
```

### 测试
运行`test.py`来评估训练好的模型。  
参数: 
```
optional arguments:
  --test_data_file TEST_DATA_FILE
                        Test data file path
  --run_dir RUN_DIR     Restore the model from this run
  --checkpoint CHECKPOINT
                        Restore the graph from this checkpoint
  --batch_size BATCH_SIZE
                        Test batch size
```
您可以运行`test.py`开始评估。例如:
```
python test.py --test_data_file=data/test_data.csv --prediction_dir=prediction --model_dir=model/1600693479 --params_dir=params
```
