import json
import copy
import string
import random


# from zhon.hanzo import punctuation

punc = ['！', '？', '｡', '＂',
        '，', ',', '：', '“',
        '、', '．', '”', '?', ':',
        '(', ')', '·']

f = open("../data/train_gcrc_replace.json", 'r')

content = f.read()
a = json.loads(content)

num = 0.
tnum = 0.

random.seed(941012)

fo = open("data/train_gcrc_aug.json", 'w', encoding='utf-8')

num = 0.

adict={0:'A', 1:'B', 2:'C', 3:'D'}

corpus = []
fo.write('[\n')

augcorpus = []

term_list = ['。', '，', '？', '；']


def split_content(contents):
    start = 0
    content_list = []
    for i in range(len(contents)//block_len):
        content_list.append(contents[start:start+block_len])
        start += block_len
    content_list.append(contents[start:])
    return content_list

def split_content_sentence(contents):
    content = ''
    content_list = []
    cur_idx = 0
    for idx, i in enumerate(contents):
        if i in term_list:
            content += i
            content_list.append(content)
            content = ''
            cur_idx = idx
        else:
            content += i
    if cur_idx < idx:
        content_list.append(contents[cur_idx+1:])
    return content_list

def split_content_sentence_lemon(contents):
    import re
    pattern = r'\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = re.split(pattern, contents)

    result_list = [i for i in result_list if len(list(i)) >= 8]

    return result_list

content_corpus = {}

for sample in a:

    if random.randint(0, 100) >= 20:

        for qidx, q1 in enumerate(sample['Questions']):
            for cidx, c in enumerate(q1['Choices']):
                sample['Questions'][qidx]['Choices'][cidx] = adict[cidx] + '. ' + c
        corpus.append(json.dumps(sample, indent=4, ensure_ascii=False))
        continue

    truesample = copy.deepcopy(sample)
    falsesample = copy.deepcopy(sample)

    article = sample['Content']

    content_list = split_content_sentence(article.replace('\t', ' ').replace('\n', ' '))

    if article not in content_corpus:
        content_corpus[article] = content_list

    # content = list(sample['Content'])

    for qidx, q1 in enumerate(sample['Questions']):

        if ("不正确" in q1['Question'] or '不符合' in q1['Question']):
            falsesample['Questions'] = [falsesample['Questions'][0]]
            falsesample['Questions'][0]['Question'] = q1['Question'].replace('不正确', '正确').replace('不符合', '符合')
            falsesample['Questions'][0]['Choices'] = []

            rcidx = random.randint(0, 3)

            # if adict[rcidx] == q1['Answer']:
            #    rcidx = 0

            falsesample['Questions'][0]['Answer'] = adict[rcidx]

            if len(content_corpus) <= 1:
                continue

            else:
                for cidx, c in enumerate(q1['Choices']):
                    if cidx != rcidx:
                        article_idx = random.randint(0, len(content_corpus)-1)

                        if list(content_corpus.keys())[article_idx] != article:
                            #print('yoyo')
                            randomcontentlist = content_corpus[list(content_corpus.keys())[article_idx]]
                            sentence_idx = random.randint(0, len(randomcontentlist)-1)
                            answer = adict[cidx] + '. ' + randomcontentlist[sentence_idx]
                        else:
                            answer = 'none'

                        falsesample['Questions'][0]['Choices'].append(answer)
                    else:
                        falsesample['Questions'][0]['Choices'].append(adict[cidx] + '. ' + q1['Choices'][cidx])

            corpus.append(json.dumps(falsesample, indent=4, ensure_ascii=False))
        elif ("正确" in q1['Question'] or '符合' in q1['Question']):

            truesample['Questions'] = [truesample['Questions'][0]]
            truesample['Questions'][0]['Question'] = q1['Question'].replace('正确', '不正确').replace('符合', '不符合')
            truesample['Questions'][0]['Choices'] = []

            rcidx = random.randint(0, 3)

            # if adict[rcidx] == q1['Answer']:
            #    rcidx = 0

            truesample['Questions'][0]['Answer'] = adict[rcidx]

            for cidx, c in enumerate(q1['Choices']):
                if cidx != rcidx:
                    sentence_idx = random.randint(0, len(content_list)-1)

                    truesample['Questions'][0]['Choices'].append(adict[cidx] + '. ' + content_list[sentence_idx])
                else:
                    truesample['Questions'][0]['Choices'].append(adict[cidx] + '. ' + q1['Choices'][cidx])

            corpus.append(json.dumps(truesample, indent=4, ensure_ascii=False))

fo.write(',\n'.join(corpus))

fo.write('\n]')

f.close()
fo.close()