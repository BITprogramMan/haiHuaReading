import json
import copy

f = open("data/train_process.json", 'r')

fo = open("data/train_aug_select_content.json", 'w', encoding='utf-8')

datas = json.loads(f.read())

adict={0:'A', 1:'B', 2:'C', 3:'D'}

corpus = []
fo.write('[\n')

dicts = ['下列、', '下列', '下面', '阅读', '（',' ）', '一项', '【', '】',
         '小题', '(',  ')',  '）']

augcorpus = []

for sample in datas:
    outsample = copy.deepcopy(sample)

    content = list(sample['Content'])

    for qidx, q1 in enumerate(sample['Questions']):

        if len(content) >= 512:

            choices = set(list(''.join(q1['Choices']).replace(' ', '')))

            maxl = 0
            maxi = 0
            for i in range(0, len(content) - 512):
                l = len(set(content[i : i + 512]) & choices)
                if l > maxl:
                    maxl = l
                    maxi = i

            outsample['Content'] = ''.join(content[maxi:maxi+512])

            outsample['Questions'] = [q1]

            corpus.append(json.dumps(outsample, indent=4, ensure_ascii=False))

fo.write(',\n'.join(corpus))

fo.write('\n]')

f.close()
fo.close()