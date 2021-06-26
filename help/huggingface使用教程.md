### BertConfig 

+ **transformers.BertConfig** 可以自定义 Bert 模型的结构，参数都是可选的

```python
from transformers import BertModel, BertConfig

configuration = BertConfig() # 进行模型的配置，变量为空即使用默认参数

model = BertModel(configuration) # 使用自定义配置实例化 Bert 模型

configuration = model.config # 查看模型参数

BertConfig.from_pretrained(pretrained_model_name_or_path,cache_dir )
# pretrained_model_name_or_path有三种情况，第一种是huggingface库中可以查到的model name，第二种情况是本地文件夹，文件夹中包含configuration file，第三种情况是直接是json文件
# cache_dir 是可选参数，指定下载的配置文件地址
```

+ **BertTokenizer**

```python
bertTokenizer=BertTokenizer(vocab_file=path,do_lower_case=True, do_basic_tokenize=True, never_split=None, unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]', tokenize_chinese_chars=True, strip_accents=None)
# path为本地字典文件路径
bertTokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path:)
```

