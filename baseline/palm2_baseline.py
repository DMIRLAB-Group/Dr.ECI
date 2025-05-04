import os
import json
import time
import google.generativeai as palm
from tqdm import tqdm
import prompt_causal_baseline

os.environ["HTTP_PROXY"] = "http://127.0.0.1:64094"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:64094"
api_key = 'AIzaS'
palm.configure(api_key=api_key)

model_params = {
    "model": "models/text-bison-001",
    # "version": "001",
    "temperature": 0.0,
    # "inputTokenLimit": 8196,
    # "outputTokenLimit": 1024,
    "candidate_count": 1,
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_DEROGATORY",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        },
        {
            "category": "HARM_CATEGORY_TOXICITY",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        },
        {
            "category": "HARM_CATEGORY_SEXUAL",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        },
        {
            "category": "HARM_CATEGORY_VIOLENCE",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        },
        {
            "category": "HARM_CATEGORY_MEDICAL",  # Adjust this according to the API's expected format
            "threshold": "BLOCK_NONE"  # Adjust this according to the API's expected format
        }
    ]
    # "top_p": 0.95,
    # "top_k": 40
}

def make_request(prompt):
    while True:
        try:
            response = palm.generate_text(prompt=prompt, **model_params)
            if hasattr(response, 'result'):
                return response.result
            else:
                print("Error: Unsuccessful response")
                time.sleep(3)

        except Exception as e:
            print("Error during request: ", e)
            time.sleep(3)


def process_item(item):
    text = item["text"]
    text = text.replace("causal </s></s> unrelated </s></s>", "")
    source = item.get('source', '')
    target = item.get('target', '')

    formal_prompt = prompt_causal_baseline.formal
    input_eci = f"Text: {text}\n Source: {source}\n Target: {target}\n"
    base_question_prompt = prompt_causal_baseline.base_question(source, target)
    # item["relationship"] = make_request(text + "\n" + source + "\n" + target + "\n" + relationship)

    cot_prompt = prompt_causal_baseline.cot(source, target)
    # item['cot_pre'] = make_request(input_eci + cot_prompt)
    # cot_history_prompt = f"# Question:{input_eci + cot_prompt}\n<response>{item['cot_pre']}<response>\n\n# {formal_prompt}\n"
    # item['cot_pre_formal'] = make_request(cot_history_prompt)
#cot one-shot
    one_shot_cot_prompt = prompt_causal_baseline.one_shot_cot_prompt
    item['one_shot_cot_pre'] = make_request(one_shot_cot_prompt + "###\n" + input_eci + cot_prompt)
    one_shot_cot_history_prompt = f"# Question:{input_eci + cot_prompt}\n<response>{item['one_shot_cot_pre']}<response>\n\n# {formal_prompt}\n"
    item['one_shot_cot_pre_formal'] = make_request(one_shot_cot_history_prompt)

# #self_ask
#     self_ask_prompt = prompt_causal_baseline.self_ask(source, target)
#     item['self_ask_pre'] = make_request(input_eci + self_ask_prompt)
#     self_ask_history_prompt = f"### Question:{input_eci + self_ask_prompt}\n<response>{item['self_ask_pre']}<response>\n\n### {formal_prompt}\n"
#     item['self_ask_pre_formal'] = make_request(self_ask_history_prompt)
#one_shot_self_ask
    # one_shot_self_ask_prompt = prompt_causal_baseline.one_shot_self_ask_prompt
    # item['one_shot_self_ask_pre'] = make_request(input_eci + base_question_prompt + one_shot_self_ask_prompt)
    # one_shot_self_ask_history_prompt = f"###<response>{item['one_shot_self_ask_pre']}<response>\n\n### {formal_prompt}\n"
    # item['one_shot_self_ask_pre_formal'] = make_request(one_shot_self_ask_history_prompt)



# #least_to_most_decomposer
#     least_to_most_decomposer_prompt = prompt_causal_baseline.least_to_most_decomposer(source, target, text)
#     item['least_to_most_decomposer_pre'] = make_request(input_eci + base_question_prompt + least_to_most_decomposer_prompt)
#     least_to_most_subq_solver_prompt = prompt_causal_baseline.least_to_most_subq_solver(source, target, text)
#     least_to_most_decomposer_history_prompt = f"### Question:{input_eci + base_question_prompt + least_to_most_decomposer_prompt}\n<response>{item['least_to_most_decomposer_pre']}<response>\n\n### {least_to_most_subq_solver_prompt}\n"
#     item['least_to_most_subq_solver_pre'] = make_request(least_to_most_decomposer_history_prompt)
#
#     least_to_most_subq_solver_history_prompt = f'''###<response>{item['least_to_most_decomposer_pre']}<response>\n\n###
#     <response>{item['least_to_most_subq_solver_pre']}<response>\n\n### {formal_prompt}\n'''
#     item['least_to_most_pre'] = make_request(least_to_most_subq_solver_history_prompt)

    return item


input_file_path = './data/data/causal_timebank.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

processed_data = []
for item in tqdm(data, desc="Processing", unit="item"):
    processed_data.append(process_item(item))

output_file_path = './0413baseline/ctb_cot_8-shot.json'

with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file_path}")
