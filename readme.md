![image-20210623162306050](figure/image-20210623162306050.png)

## 2021海华AI挑战赛·中文阅读理解·技术组

> 《2021海华AI挑战赛·中文阅读理解》大赛由中关村海华信息技术前沿研究院与清华大学交叉信息研究院联合主办，腾讯云计算协办。共设置题库16000条数据，总奖金池30万元，比赛的数据主要来自中学语文考试的阅读理解题库。每条数据都包括 1 篇文章（content），至少 1 个问题（question）和 2 - 5 个候选选项（choices）。比赛模型需要根据文章和问题，选择正确的选项。

+ [Dataset（密码：8mxi）](https://pan.baidu.com/s/1Fm97-1SVDyFbDT6s95zY0Q )

+ 这个项目总结了我的比赛经验，参考了冠军队伍的比赛代码。想要仔细了解冠军队伍思路，参考这里[（密码：fulk）](https://pan.baidu.com/s/19wOWeCZkDodsai-GHE6LyQ)

### 数据分析与可视化

#### 一条样例数据如下：

```json
{
    "ID": "0001",
    "Content": "春之怀古张晓风春天必然曾经是这样的：从绿意内敛的山头，一把雪再也撑不住了，噗嗤的一声，将冷面笑成花面，一首澌澌然的歌便从云端唱到山麓，从山麓唱到低低的荒村，唱入篱落，唱入一只小鸭的黄蹼，唱入软溶溶的春泥——软如一床新翻的棉被的春泥。  那样娇，那样敏感，却又那样浑沌无涯。一声雷，可以无端地惹哭满天的云，一阵杜鹃啼，可以斗急了一城杜鹃花。一阵风起，每一棵柳都会吟出一则白茫茫、虚飘飘说也说不清、听也听不清的飞絮，每一丝飞絮都是一株柳的分号。反正，春天就是这样不讲理，不逻辑，而仍可以好得让人心平气和的。 春天必然曾经是这样的：满塘叶黯花残的枯梗抵死苦守一截老根，北地里千宅万户的屋梁受尽风欺雪扰自温柔地抱着一团小小的空虚的燕巢。然后，忽然有一天，桃花把所有的山村水廓都攻陷了。柳树把皇室的御沟和民间的江头都控制住了——春天有如旌旗鲜明的王师，因为长期虔诚的企盼祝祷而美丽起来。 而关于春天的名字，必然曾经有这样的一段故事：在《诗经》之前，在《尚书》之前，在仓颉造字之前，一只小羊在啮草时猛然感到的多汁，一个孩子放风筝时猛然感觉到的飞腾，一双患风痛的腿在猛然间感到舒适，千千万万双素手在溪畔在江畔浣纱时所猛然感到的水的血脉……当他们惊讶地奔走互告的时候，他们决定将嘴噘成吹口哨的形状，用一种愉快的耳语的声音来为这季节命名——“春”。 鸟又可以开始丈量天空了。有的负责丈量天的蓝度，有的负责丈量天的透明度，有的负责用那双翼丈量天的高度和深度。而所有的鸟全不是好的数学家，他们吱吱喳喳地算了又算，核了又核，终于还是不敢宣布统计数字。 至于所有的花，已交给蝴蝶去数。所有的蕊，交给蜜蜂去编册。所有的树，交给风去纵宠。而风，交给檐前的老风铃去一一记忆，一一垂询。 春天必然曾经是这样，或者，在什么地方，它仍然是这样的吧？穿越烟囱与烟囱的黑森林，我想走访那踯躅在湮远年代中的春天。",
    "Questions": [
      {
        "Q_id": "000101",
        "Question": "鸟又可以开始丈量天空了。”这句话的意思是   （   ）",
        "Choices": [
          "A．鸟又可以飞了。",
          "B． 鸟又要远飞了。",
          "C．鸟又可以筑巢了。"
        ],
        "Answer": "A"
      },
      {
        "Q_id": "000102",
        "Question": "本文写景非常含蓄，请读一读找一找哪些不在作者的笔下有所描述",
        "Choices": [
          "A．冰雪融化",
          "B． 蝴蝶在花间飞舞",
          "C．白云在空中飘",
          "D．小鸟在空中自由地飞"
        ],
        "Answer": "C"
      }
```

### 数据增强

+ 方案一：

​        对于题目中包含“正确” 或“符合” 的题目，将题目中的“正确” 替换成“不正确” （“符合” 替换成“不符合” ），保留原来为答案的选项，再从剩下的三个选项中随机选择一个选项作为该题的答案。构造二选一的反向题。类似，对于题目中包含“不正确” 或“不符合” 的题目，将题目中的“不正确” 替换成“正确” （“不符合” 替换成“符合” ），保留原来为答案的选项，再从剩下的三个选项中随机选择一个选项作为该题的答案。构造二选一的反向题。

+ 方案二：

​           对于 content 的字符长度大于 512 的训练样例，选择和选项 choice 重合度最大的长度为 512 的片段作为这个样例的 content。

+ 方案三：

​          抽取出训练集中的断句题用于古文 bert，并抽取测试集中的断句题。

+ 方案四：

​        **数据扩充**

​        c3:https://github.com/nlpdata/c3

​        gcrc:https://github.com/jfzy-lab/GCRC

### Model

+ BertForMultipleChoice
+ ElectraForMultipleChoice

使用对抗训练的方法：FGM与PGD，关于对抗训练的介绍参考[这里](https://github.com/BITprogramMan/haiHuaReading/blob/master/%E5%AF%B9%E6%8A%97%E8%AE%AD%E7%BB%83.md)

```python
import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

```

+ 预训练模型的选择:

BERT:https://huggingface.co/hfl/chinese-bert-wwm-ext

RoBERT:https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

MacBERT:https://huggingface.co/hfl/chinese-macbert-large

Guwen-BERT:https://github.com/Ethan-yt/guwenbert

ALBERT:https://huggingface.co/voidful/albert_chinese_xxlarge

### 评价指标

![image-20210626152237706](figure/image-20210626152237706.png)

### 运行

+ main.py给出baseline模型的框架，需要根据具体数据以及模型修改配置参数。































