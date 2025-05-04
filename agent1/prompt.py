##explorer
def causal(source, target, text):
    prompt = f'''
"text": {text}
"source event": {source},
"target event": {target},

Question: Is there a causal relationship between {source} and {target} ? 

    '''
    return prompt

def causation(source, target, text):
    prompt = f'''
"text": {text}
"source event": {source},
"target event": {target},
Question: Is there causation between {source} and {target} ? 

    '''
    return prompt

def causality(source, target, text):
    prompt = f'''
"text": {text}
"source event": {source},
"target event": {target},

Question: Is there causality between {source} and {target} ? 

    '''
    return prompt

system = '''
You are an expert at identifying cause-effect relationships between events in a text. Please answer this question based on the context given to you.
'''

vanilla_formal = "Your final answer should be {{\"Answer\":\"Yes\"}} or {{\"Answer\":\"No\"}}, in the format of JSON {{\"Answer\":\" \"}}, at the end of your reply."
