import json
def load_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

file_path = '../agent2/output/mediator_ds.jsonl'
data = load_json_lines(file_path)

def extract_mediator_ans_formal(mediator_ans):

    start_index = mediator_ans.find('Answer to subquestion 4: ') + len('Answer to subquestion 4: ')
    end_index = mediator_ans.find('\n', start_index)

    if start_index != -1:
        if end_index == -1:
            end_index = len(mediator_ans)

        return mediator_ans[start_index:end_index].strip()
    return None

for item in data:
    mediator_ans = item.get('mediator_ans', '')
    mediator_ans_formal = extract_mediator_ans_formal(mediator_ans)
    item['mediator_ans_formal'] = mediator_ans_formal

output_file_path = './output/get_mediator.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as file:
    for obj in data:
        json.dump(obj, file)
        file.write('\n')

