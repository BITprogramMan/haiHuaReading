import json
import copy

f = open("data/train_process.json", 'r')

fo = open("data/train_augtrue.json", 'w', encoding='utf-8')

datas = json.loads(f.read())

adict={0:'A', 1:'B', 2:'C', 3:'D'}

corpus = []
fo.write('[\n')

augcorpus = []

for sample in datas:
    truesample = copy.deepcopy(sample)

    content = list(sample['Content'])

    for qidx, q1 in enumerate(sample['Questions']):

        if ("不正确" in q1['Question'] or '不符合' in q1['Question']):
            continue
        elif ("正确" in q1['Question'] or '符合' in q1['Question']):
            truesample['Questions'] = [truesample['Questions'][0]]
            truesample['Questions'][0]['Question'] = q1['Question'].replace('正确', '不正确').replace('符合', '不符合')
            truesample['Questions'][0]['Choices'] = []

            import random

            rcidx = random.randint(1, 3)

            if adict[rcidx] == q1['Answer']:
                rcidx = 0

            truesample['Questions'][0]['Answer'] = adict[rcidx]

            for cidx, c in enumerate(q1['Choices']):
                if adict[cidx] != q1['Answer'] and cidx != rcidx:
                    truesample['Questions'][0]['Choices'].append('none')
                else:
                    truesample['Questions'][0]['Choices'].append(q1['Choices'][cidx])

            corpus.append(json.dumps(truesample, indent=4, ensure_ascii=False))

fo.write(',\n'.join(corpus))

fo.write('\n]')

f.close()
fo.close()