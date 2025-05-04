import os
import json
import time
import requests
from tqdm import tqdm
import prompt
import openai
import re

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

openai.api_key = 'sk-'

def gpt(messages, model='gpt-4o-mini', response_length=16384, temperature=0, top_p=0, frequency_penalty=0,
        presence_penalty=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=response_length,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response


def decoder_for_gpt(messages, response_length=16384, temperature=0):
    response = gpt(messages, response_length=response_length, temperature=temperature)
    return response["choices"][0]["message"]["content"]


def is_connected():
    try:
        requests.get('https://www.google.com/', timeout=3)
        return True
    except requests.ConnectionError:
        return False
input_file_path = '../agent2/mediator.jsonl'

# input_file_path = './baseline/gpt_4o/Dr.ECI/ESC_ECI.sample1000_300_mediator_BOTH.jsonl'
# input_file_path = '../data/gpt4_data/ESC_ECI.sample1000.jsonl'
with open(input_file_path, 'r', encoding='utf-8') as f:
    representative_cots_data = [json.loads(line.strip()) for line in f]

def generate_cot_text(item):
    words = item.get('words', [])
    events = item.get('events', [])
    source_indices = events[0]
    target_indices = events[1]
    bi_causal_label = item.get('bi_causal_label', [])
    sample = item.get('sample', [])
    ground_true = 'Causal' if bi_causal_label == 1 else 'Non-Causal'

    source_event = " ".join([words[t] for t in source_indices])
    target_event = " ".join([words[t] for t in target_indices])

    words_with_tags = words[:]
    for idx in source_indices:
        words_with_tags[idx] = f"<source event>{words_with_tags[idx]}<source event>"
    for idx in target_indices:
        words_with_tags[idx] = f"<target event>{words_with_tags[idx]}<target event>"

    text = " ".join(words_with_tags)

    mediator_both = item.get('mediator_both', [])

    llm_pre = item.get('REans', [])
    return text, source_event, target_event, ground_true, llm_pre, mediator_both,sample
def calculate_cost(response):
    usage = response['usage']
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)

    input_cost = (prompt_tokens / 1_000_000) * 0.150
    output_cost = (completion_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost

    return total_cost

total_cost = 0.0
results = []
start_time = time.time()

for index, item in tqdm(enumerate(representative_cots_data), desc="Processing items", unit=" item"):
    try:

        while not is_connected():
            print("No internet connection. Retrying in 60 seconds...")
            time.sleep(60)

        text, source_event, target_event, ground_true, llm_pre, mediator_both, sample = generate_cot_text(item)
      # ## reasoner
        reasoner_system = prompt.reasoner_system
        reasoner = prompt.reasoner(source_event, target_event, text, mediator_both)
        messages = [
            {"role": "system", "content": reasoner_system},
            {"role": "user", "content": reasoner}
        ]
        print(messages)

        response_text = decoder_for_gpt(messages, response_length=1000, temperature=0)
        print(response_text)

        response = gpt(messages, response_length=16384, temperature=0)
        cost = calculate_cost(response)
        total_cost += cost

        results.append({
            "events_id": item["events_id"],
            "words": item["words"],
            "tri_causal_label": item["tri_causal_label"],
            "bi_causal_label": item["bi_causal_label"],
            "events": item["events"],

            'reasoner_ans': response_text,
            'cost': total_cost
        })
        print(f"Item {index} processed successfully. Cost: ${total_cost:.6f}")

    except Exception as e:
        print(f"Error processing data for item {index}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total cost for processing items: ${total_cost:.6f}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")

output_file_path = 'reasoner.jsonl'

with open(output_file_path, 'w', encoding='utf-8') as f:
    for result in results:
        json_line = json.dumps(result, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Results saved to {output_file_path}")
