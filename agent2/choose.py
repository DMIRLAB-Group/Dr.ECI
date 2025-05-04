
import json
import re
from tqdm import tqdm

input_file_path = '../agent1/output/explorer_ds.jsonl'


C_correct = 0
C_Answer = 0
C_bi_causal_label = 0
C_Answer_data = []
total = 0
C_correct_data = []


def contains_yes_answer(text):
    rules = [
        r'{"Answer":"Yes"}',
        r'{\n"Answer": "Yes"\n}'

    ]

    if text is not None:
        for rule in rules:
            if re.search(rule, text):
                return True
    return False



with open(input_file_path, 'r', encoding='utf-8') as file, \
     open('./output/explorer_choose.jsonl', 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(tqdm(file, desc="Processing", unit="item")):
        data = json.loads(line)

        total_yes_count = sum(contains_yes_answer(data.get(rule, '')) for rule in
                            ["causation_ans", "causal_ans", "causality_ans"])

        if total_yes_count >= 1 and data["bi_causal_label"] == 1:
            C_correct += 1
            C_correct_data.append(data)
        if total_yes_count >= 1:
            C_Answer += 1
            outfile.write(json.dumps(data) + '\n')



        if data["bi_causal_label"] == 1:
            C_bi_causal_label += 1
        total += 1

P = C_correct / C_Answer if C_Answer != 0 else 0
R = C_correct / C_bi_causal_label if C_bi_causal_label != 0 else 0
F = 2 * P * R / (P + R) if P + R != 0 else 0

print("Precision (P):", P)
print("Recall (R):", R)
print("F1 Score (F):", F)

print("C_bi_causal_label =:", C_bi_causal_label)
print("C_Answer =:", C_Answer)
print("C_correct =:", C_correct)
print("tolte =:", total)

