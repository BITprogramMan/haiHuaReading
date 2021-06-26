
import json
import copy

fin = open("data/dev_process.json", 'r')
fo = open("data/dev_dj.json", 'w')

datas = json.loads(fin.read())

corpus = []

fo.write('[\n')

for sample in datas:

    segsample = copy.deepcopy(sample)

    segsample['Questions'] = []

    article = list(sample['Content'])

    for q in sample['Questions']:
        if '断句' in q['Question'] \
            or '停顿' in q['Question'] \
            or '朗读' in q['Question']:

            for cidx, c in enumerate(q['Choices']):
                q['Choices'][cidx] = c.replace('／', '/').replace('\\', '/')

            segsample['Questions'].append(q)

            corpus.append(json.dumps(segsample, indent=4, ensure_ascii=False))

fo.write(',\n'.join(corpus))
fo.write('\n]')

fin.close()
fo.close()

