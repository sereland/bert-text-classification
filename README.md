# BERT文本分类与回归项目

基于transformers使用BERT类模型进行文本分类和回归的完整项目。

## 项目结构

```
├── data
│   ├── train.csv          # 训练数据
│   └── test.csv           # 测试数据
├── models
│   ├── __init__.py
│   ├── bert.py            # BERT模型实现
│   └── bert_cnn.py        # BERT-CNN模型实现
├── utils
│   ├── __init__.py
│   ├── bert_utils.py      # BERT训练工具
│   └── data_utils.py      # 数据处理工具
├── config.py              # 配置文件
├── train.py               # 训练脚本
└── README.md              # 项目说明
```

## 功能特点

- **多种模型支持**：支持基础BERT和BERT-CNN模型
- **多任务支持**：支持分类和回归任务
- **灵活配置**：通过配置文件轻松调整模型和训练参数
- **完整工具链**：包含数据处理、模型训练、评估和预测
- **早停机制**：支持早停防止过拟合
- **多种优化器**：支持AdamW、Adam、SGD等优化器
- **学习率调度**：支持线性、余弦、常数调度器

## 环境要求

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 数据格式

数据文件应包含以下字段：
- `query`：查询文本
- `text`：正文文本
- `label`：标签（分类任务为类别ID，回归任务为数值）

示例：
```csv
query,text,label
如何学习机器学习,机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测或决策。,1
Python编程入门,Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。,0
```

## 使用方法

### 1. 训练模型

#### 分类任务
```bash
python train.py --do_train --task_type classification --model_type bert --num_epochs 3
```

#### 回归任务
```bash
python train.py --do_train --task_type regression --model_type bert --num_epochs 3
```

#### 使用BERT-CNN模型
```bash
python train.py --do_train --task_type classification --model_type bert_cnn --num_epochs 3
```

### 2. 评估模型
```bash
python train.py --do_eval --model_path checkpoints/final_model.pt
```

### 3. 预测
```bash
python train.py --do_predict --model_path checkpoints/final_model.pt
```

## 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task_type` | 任务类型 (classification/regression) | classification |
| `--model_type` | 模型类型 (bert/bert_cnn) | bert |
| `--model_name` | 预训练模型名称 | bert-base-chinese |
| `--batch_size` | 批次大小 | 32 |
| `--learning_rate` | 学习率 | 2e-5 |
| `--num_epochs` | 训练轮数 | 3 |
| `--max_length` | 文本最大长度 | 128 |
| `--device` | 设备类型 (cuda/cpu) | cuda |
| `--early_stopping` | 是否使用早停 | False |
| `--patience` | 早停耐心值 | 3 |

## 配置文件

[`config.py`](config.py) 包含了所有配置参数，可以通过修改配置文件或命令行参数来调整：

- **数据配置**：数据路径、文本列名、标签列名
- **模型配置**：模型类型、预训练模型名称、最大长度、类别数
- **训练配置**：任务类型、批次大小、学习率、训练轮数
- **优化器配置**：优化器类型、权重衰减、预热比例
- **设备配置**：设备类型
- **保存配置**：保存目录、保存步数、评估步数
- **早停配置**：早停开关、耐心值
- **BERT-CNN配置**：CNN滤波器、卷积核大小、dropout率

## 模型说明

### BERT模型
- [`BertClassifier`](models/bert.py)：BERT分类器，使用[CLS] token的表示进行分类
- [`BertRegressor`](models/bert.py)：BERT回归器，使用[CLS] token的表示进行回归
- [`BertMultiTask`](models/bert.py)：BERT多任务模型，同时支持分类和回归

### BERT-CNN模型
- [`BertCNNClassifier`](models/bert_cnn.py)：BERT-CNN分类器，结合BERT和CNN特征
- [`BertCNNRegressor`](models/bert_cnn.py)：BERT-CNN回归器，结合BERT和CNN特征
- [`BertCNNGated`](models/bert_cnn.py)：带门控机制的BERT-CNN模型，融合CNN和[CLS]特征

## 工具说明

### 数据处理工具
- [`DataProcessor`](utils/data_utils.py)：数据加载、预处理和分割
- [`TextDataset`](utils/data_utils.py)：PyTorch数据集类
- [`create_data_loaders`](utils/data_utils.py)：创建训练和验证数据加载器
- [`create_test_dataloader`](utils/data_utils.py)：创建测试数据加载器

### 训练工具
- [`ModelTrainer`](utils/bert_utils.py)：模型训练器，包含训练循环和验证
- [`ModelPredictor`](utils/bert_utils.py)：模型预测器，支持单文本和批量预测
- [`MetricsCalculator`](utils/bert_utils.py)：指标计算器，支持分类和回归指标
- [`EarlyStopping`](utils/bert_utils.py)：早停机制

## 训练过程

1. **数据加载**：自动加载训练和测试数据
2. **数据预处理**：文本合并、标签编码、数据分割
3. **模型初始化**：根据配置创建相应模型
4. **训练循环**：
   - 前向传播计算损失
   - 反向传播更新参数
   - 定期验证和早停检查
   - 保存最佳模型
5. **模型评估**：在测试集上评估模型性能
6. **结果保存**：保存模型、配置和训练历史

## 输出文件

训练完成后，`checkpoints`目录将包含：
- `final_model.pt`：最终训练的模型
- `best_model_epoch_X.pt`：验证集上表现最好的模型
- `config.json`：训练配置
- `training_history.json`：训练历史记录
- `evaluation_results.json`：评估结果

## 示例数据

项目包含示例数据：
- [`data/train.csv`](data/train.csv)：500条训练数据，包含AI相关文本
- [`data/test.csv`](data/test.csv)：100条测试数据

数据标签说明：
- `1`：AI/技术相关内容
- `0`：编程/开发相关内容

## 扩展功能

### 添加新模型
1. 在`models/`目录下创建新的模型文件
2. 继承`nn.Module`并实现`forward`方法
3. 在`train.py`中添加模型创建逻辑

### 添加新的数据处理方法
1. 在`utils/data_utils.py`中扩展`DataProcessor`类
2. 添加新的预处理方法
3. 更新数据加载逻辑

### 自定义训练流程
1. 在`utils/bert_utils.py`中扩展`ModelTrainer`类
2. 重写训练循环或添加新的训练策略
3. 实现自定义的评估指标

## 注意事项

1. **数据格式**：确保CSV文件包含正确的列名
2. **内存使用**：大批次训练需要足够的GPU内存
3. **模型大小**：BERT模型较大，首次运行会下载预训练模型
4. **中文支持**：默认使用`bert-base-chinese`，支持中文文本
5. **类别平衡**：分类任务建议检查类别分布是否平衡

## 常见问题

### Q: 如何处理长文本？
A: 调整`--max_length`参数，但注意GPU内存限制。

### Q: 如何使用不同的预训练模型？
A: 修改`--model_name`参数，如`--model_name bert-large-chinese`。

### Q: 如何调整学习率？
A: 使用`--learning_rate`参数，或修改配置文件中的`LEARNING_RATE`。

### Q: 如何保存更多检查点？
A: 调整`--save_steps`参数，减少保存间隔。

### Q: 如何使用多GPU训练？
A: 项目暂不支持多GPU训练，需要自行修改训练脚本。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过GitHub Issues联系。