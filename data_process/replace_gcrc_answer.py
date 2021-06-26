import json
dict_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
with open('ensemble.3.gcrc.txt', 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    answer_list = []
    for line in lines:
        score_list = [float(s) for s in line.strip().split('\t')]
        max_index = score_list.index(max(score_list))
        # print(dict_answer[max_index])
        answer_list.append(dict_answer[max_index])

print(answer_list)
print(len(answer_list))
train_replace = []
train_gcrc_right = []
train_gcrc_wrong = []
total_num = 0
correct_num = 0
with open('train_gcrc.json', 'r', encoding='utf-8') as fin:
    data_raw = json.load(fin)
    for d in data_raw:
        for q in d['Questions']:
            if q['Answer'] == answer_list[total_num]:
                correct_num += 1
                train_gcrc_right.append(d)
            else:
                train_gcrc_wrong.append(d)
            q['Answer'] = answer_list[total_num]
            total_num += 1
        train_replace.append(d)

print(total_num)
print(correct_num/total_num)

with open('train_gcrc_right.json', 'w', encoding='utf-8') as f:
    json.dump(train_gcrc_right, f, ensure_ascii=False, indent=1)

with open('train_gcrc_wrong.json', 'w', encoding='utf-8') as f:
    json.dump(train_gcrc_wrong, f, ensure_ascii=False, indent=1)

with open('train_gcrc_replace.json', 'w', encoding='utf-8') as f:
    json.dump(train_replace, f, ensure_ascii=False, indent=1)