import json
import copy
import jieba
# import jiayan
# from jiayan import load_lm
# from jiayan import CRFPunctuator
import re
f = open("../data/train.json", 'r')
fo = open("data/train_process.json", 'w', encoding='utf-8')

datas = json.loads(f.read())



corpus = []
fo.write('[\n')

adict={0:'A', 1:'B', 2:'C', 3:'D'}

dicts = ['下列、', '下列', '下面', '阅读', '（',' ）', '一项', '【', '】',
         '小题', '(',  ')',  '）']

for sample in datas:

    # process question and choices

    for qidx, q in enumerate(sample['Questions']):

        for j in dicts:
            sample['Questions'][qidx]['Question'] = q['Question'].replace(j, '')

        sample['Questions'][qidx]['Question'] = re.sub(r'[0-9].', '', q['Question'])

        for idx in range(len(q['Choices'])):
            q['Choices'][idx] = q['Choices'][idx].replace('．', '').replace('、', '').replace(' ','') \
                .replace('.','').replace('A', 'A. ').replace('B', 'B. ').replace('C', 'C. ').replace('D', 'D. ')


    corpus.append(json.dumps(sample, indent=4, ensure_ascii=False))

fo.write(',\n'.join(corpus))

fo.write('\n]')

f.close()
fo.close()