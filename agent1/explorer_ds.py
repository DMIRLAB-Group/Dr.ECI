import os
import json
import time
import requests
from tqdm import tqdm
import prompt
from openai import OpenAI


os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com")

def gpt(messages, model='deepseek-chat', response_length=8192, temperature=1, top_p=0.7, frequency_penalty=0,
        presence_penalty=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=response_length,
        # response_format={
        #     'type': 'json_object'
        # },
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response

def decoder_for_gpt(messages, response_length=8192, temperature=1):
    response = gpt(messages, response_length=response_length, temperature=temperature)

    return response.choices[0].message.content
def is_connected():
    try:
        requests.get('https://www.google.com/', timeout=3)
        return True
    except requests.ConnectionError:
        return False

input_file_path = '../data/ESC/intra_sent_causality.json'
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

    llm_pre = item.get('addcon_ans', [])
    return text, source_event, target_event, ground_true, llm_pre, mediator_both,sample

total_cost = 0.0
results = []
start_time = time.time()


for index, item in tqdm(enumerate(representative_cots_data[:2]), desc="Processing items", unit=" item"):
    try:

        while not is_connected():
            print("No internet connection. Retrying in 60 seconds...")
            time.sleep(60)

        text, source_event, target_event, ground_true, llm_pre, mediator_both, sample = generate_cot_text(item)

        # Round 1
        system_content = prompt.system
        causal = prompt.causal(source_event, target_event, text)
        causality = prompt.causality(source_event, target_event, text)
        causation = prompt.causation(source_event, target_event, text)
        ans_formal = prompt.vanilla_formal

        messages1 = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": causal + ans_formal}
        ]
        print(f"Messages Round 1: {messages1}")

        response = gpt(messages1, response_length=8192, temperature=1)
        response_text_round1 = response.choices[0].message.content
        print(response_text_round1)
        messages1.append(response.choices[0].message)

        # Round 2
        messages2 = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": causation + ans_formal}
        ]
        print(f"Messages Round 2: {messages2}")
        response = gpt(messages2, response_length=8192, temperature=1)
        response_text = response.choices[0].message.content
        print(response_text)

        messages2.append(response.choices[0].message)

        # Round 3
        messages3 = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": causality + ans_formal}
        ]
        print(f"Messages Round 3: {messages3}")

        response = gpt(messages3, response_length=8192, temperature=1)
        response_text_round3 = response.choices[0].message.content
        print(response_text_round3)

        messages3.append(response.choices[0].message)


        results.append({
            "events_id": item["events_id"],
            "words": item["words"],
            "tri_causal_label": item["tri_causal_label"],
            "bi_causal_label": item["bi_causal_label"],
            "events": item["events"],
            # "addcon_ans": llm_pre,
            # "addcon_ans_formal": response_text,
            "causal_ans": response_text_round1,
            'causation_ans': response_text,
            "causality": response_text_round3
        })
        print(f"Item {index} processed successfully.")

    except Exception as e:
        print(f"Error processing data for item {index}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

output_file_path = './output/explorer_ds.jsonl'
with open(output_file_path, 'w', encoding='utf-8') as f:
    for result in results:
        json_line = json.dumps(result, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Results saved to {output_file_path}")