import os
import json
import time
import google.generativeai as palm
from tqdm import tqdm
# import prompt_causal
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
api_key = 'AIz'
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
    # "topP": 0.95,
    # "topK": 40
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

    # text = re.sub(r'causal </s></s> unrelated </s></s>', '', item["text"])
    # text = item.get('text', '')
    source = item.get('source', '')
    target = item.get('target', '')

    formal_plan = prompt_causal.formal_plan
    formal_one = prompt_causal.formal_one

    rule_meditor = prompt_causal.meditor(source, target, text)
    relationship = prompt_causal.relationship
    example_prompt = prompt_causal.example()

    ###explor
    causal = prompt_causal.causal(source, target)
    causality = prompt_causal.causality(source, target)
    causation = prompt_causal.causation(source, target)

    input_eci = f"Text: {text}\n source: {source}\n target: {target}\n"
    # item["causal_prompt"] = input_eci+causal
    item["causal_ans"] = make_request(input_eci+causal)
    item["causation_ans"] = make_request(input_eci+causation)
    item["causality_ans"] = make_request(input_eci+causality)

    # one_shot_prompt = prompt_causal.one_shot(source, target, text)
    # item["relationship"] = make_request(text + "\n" + source + "\n" + target + "\n" + relationship)
    # item["rule_zero_prompt"] = rule1_content
    # item["rule_zero"] = make_request(rule_meditor)
    # explanation_prompt = prompt_causal.get_explanation_prompt()
    # # # explanation_prompt = rule2.get_explanation_prompt()
    # # history_with_new_prompt = f"# Question:{rule1_content}\n<response>{item['rule_zero']}<response>\n\n# {formal_one}\n"
    # # item["rule_zero_formal"] = make_request(history_with_new_prompt)
    # history_with_new_prompt1 = f"# Question:{rule1_content}\n<response>{item['rule_zero']}<response>\n\n# explanation: {explanation_prompt}\n"
    # item["explain"] =make_request(history_with_new_prompt1)
    return item


input_file_path = './data/all_data/causal_time_bank.json'  

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

processed_data = []
for item in tqdm(data, desc="Processing", unit="item"):
    processed_data.append(process_item(item))

# for item in processed_data:
#     print(item)

# for item in processed_data:
#     output = item.get('Output', '')
#     explanation = item.get('Explanation', '')
#     print(f"{output}\nExplanation: {explanation}\n")
output_file_path = './output_TB/palm2/explorer_CTB.json'

with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file_path}")
