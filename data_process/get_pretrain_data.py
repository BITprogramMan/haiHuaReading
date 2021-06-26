import json
import re
input_dir = 'data/dev_process.json'

dict_answer = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
set_type = 'test'
#set_type = 'test'

block_len = 150
step_len = 128
max_select = 3

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

def compute_score(content, qa):
    c_list = [i for i in content]
    qa_list = [q for q in qa]
    return jaccard(c_list, qa_list)

def jaccard(pre, ref):
    pre = set(pre)
    ref = set(ref)
    inter = pre & ref
    union = pre | ref
    if len(union) == 0:
        jaccard_score = 1
    else:
        jaccard_score = len(inter) / len(union)
    return jaccard_score

def del_char_q(src):
    a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\（.*?）", "", src)
    b = re.sub(u"[0-9]+[\.|\．|\、|\）]*", "", a)
    c = re.sub(u"一项|下列|以下|下面|的是", "", b)
    return c

def del_char_o(src):
    a = re.sub(u"[A-D]+[\.|\．|\、|\）]*", "", src)
    return a

lines = []
with open(input_dir, 'r', encoding='utf-8') as fin:
    data_raw = json.load(fin)
    # data_raw["race_id"] = file
    for d in data_raw:
        lines.append(d)

print(len(lines))

fw = open('test.pretrain.txt', "w", encoding="utf-8")
# fw_q = open('q.txt', "w", encoding="utf-8")
# fw_a = open('a.txt', "w", encoding="utf-8")

for data_raw in lines:
    id = "%s-%s" % (set_type, data_raw["ID"])
    article = data_raw["Content"]
    article = "".join(article.split())
    article_len = len(article)
    # print(article_len)
    content_list = split_content_sentence(article.replace('\t', ' ').replace('\n', ' '))

    # print(content_list)
    for index, q in enumerate(data_raw["Questions"]):
        question = q['Question']
        question = "".join(question.split())
        options = q['Choices']
        if set_type == 'test':
            truth = 0
        else:
            truth = ord(q['Answer']) - ord('A')
        del_q = del_char_q(question).strip()
        # del_q = question
        # print(del_q, file=fw_q)
        for o_id, o in enumerate(options):
            o = "".join(o.split())
            del_o = del_char_o(o).strip()
            # print(del_o, file=fw_a)
            qa = del_q + ' ' + del_o
            qa = qa.replace('\t', ' ').replace('\n', ' ')

            if o_id == truth:
                label = 1
            else:
                label = 0

            tag = id + '_' + str(index) + '_' + str(o_id)

            score_dict = {}
            # select related contents
            for idx, c in enumerate(content_list):
                score = compute_score(c, qa)
                score_dict[idx] = score

            sort_score_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
            cut_score_dict = sort_score_dict[:3]
            final_score_dict = sorted(cut_score_dict, key=lambda item: item[0])

            select_list = []
            content_len = len(content_list)
            for i in final_score_dict:
                if i[0] > 0 and i[0] < content_len -1:
                    select_list.append(i[0] - 1)
                    select_list.append(i[0])
                    select_list.append(i[0] + 1)
                elif i[0] == 0 and content_len > 1:
                    select_list.append(i[0])
                    select_list.append(i[0] + 1)
                elif i[0] == content_len -1 and content_len > 1:
                    select_list.append(i[0] - 1)
                    select_list.append(i[0])
                else:
                    select_list.append(i[0])

            # print(sort_score_dict)
            # print(final_score_dict)
            # print(select_list)
            # print(len(content_list))
            # print(list(set(select_list)))
            article_select = ''
            total_len = 0
            for idx, i in enumerate(select_list):
                total_len += len(content_list[i])
                if total_len > 500:
                    break
                article_select += content_list[i] + ' '

            #pqa = tag + '\t' + article_select.replace('\t', ' ') + '\t' + qa.replace('\t', ' ') + '\t' + str(label)
            pqa = article_select.replace('\t', ' ') + '\t' + qa.replace('\t', ' ')

            #if len(pqa.split('\t')) != 4:
            #    print(len(pqa.split('\t')))
            print(pqa.replace('\n', ' '), file=fw)

fw.close()
