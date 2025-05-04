# baseline

def base_question(source, target):
    prompt = f'''Question: Is there a causal relationship between {source} and {target}?\n'''
    return prompt

#CoT
def cot(source, target):
    prompt = f'''Question: Is there a causal relationship between {source} and {target}. Let's think step by step?'''
    return prompt

one_shot_cot_prompt = \
"""
Please answer the question by thinking step by step.
###
For example:
####
Text: Forecasters say the picture will get worse because more rains are on the way .  
Source: rains
Target: get
Question: Is there a causal relationship between rains and get, Let's think step by step?
Answer:
Step 1: The sentence says \"Forecasters say the picture will get worse because more rains are on the way\".
Step 2: The sentence implies that the rains will cause the picture to get worse.
Step 3: So there is a causal relationship between rains and get.

####
Text: The projection sent Anheuser shares plunging $ 4.375 in New York Stock Exchange composite trading yesterday .  
Source: predicting
Target: disappointments
Question: Is there a causal relationship between predicting  and disappointments, Let's think step by step?
Answer:Step 1: The projection sent Anheuser shares plunging $ 4.375 in New York Stock Exchange composite trading yesterday .\n\nStep 2: The projection is the cause of the plunging.\n\nStep 3: So the answer is causal.

####
Text: Although the government said the economy grew an estimated 2.5 % in the third quarter , in line with expectations , analysts are increasingly predicting much more sluggish growth -- and therefore more corporate earnings disappointments -- for the fourth quarter .
Source: projection
Target: plunging
Question: Is there a causal relationship between projection  and plunging, Let's think step by step?
Answer:Step 1: Predicting is the cause of disappointments.\nStep 2: The government said the economy grew an estimated 2.5 % in the third quarter , in line with expectations , analysts are increasingly predicting much more sluggish growth -- and therefore more corporate earnings disappointments -- for the fourth quarter .\nStep 3: Therefore, the final answer is causal.

####
Text: Officials in California are warning residents that oncoming rains will cause dangerous and unpredictable landslides .
Source: rains
Target: landslides
Question: Is there a causal relationship between rains  and landslides, Let's think step by step?
Answer:Step 1: Rains can cause landslides.\nStep 2: Landslides can be caused by other factors besides rains.\nStep 3: Therefore, the relationship between rains and landslides is causal.

####
Text: StatesWest Airlines , Phoenix , Ariz . , said it withdrew its offer to acquire Mesa Airlines because the Farmington , N.M., carrier did n't respond to its offer by the close of business yesterday , a deadline StatesWest had set for a response . 
Source: respond
Target: withdrew
Question: Is there a causal relationship between respond  and withdrew, Let's think step by step?
Answer:Yes. The text states that StatesWest withdrew its offer to acquire Mesa Airlines because the Farmington, N.M., carrier didn't respond to its offer by the close of business yesterday. So the response of Mesa Airlines caused StatesWest to withdraw its offer.

####
Text: That , he said , is why Unisys is overhauling its whole business : It needs to prepare for a world in which profit margins will be lower than computer companies have been used to . 
Source: needs
Target: overhauling
Question: Is there a causal relationship between needs  and overhauling, Let's think step by step?
Answer:Step 1: The source is \"needs\" and the target is \"overhauling\".\nStep 2: The source is a reason for the target.\nStep 3: The source is a cause for the target.\nStep 4: The answer is causal.

####
Text: In addition , Unisys set up an unspecified reserve -- apparently $ 60 million to $ 70 million -- to cover the minimum amount it will have to pay the government because of its involvement in the defense-procurement scandal . 
Source: involvement
Target: pay
Question: Is there a causal relationship between involvement  and pay, Let's think step by step?
Answer:Step 1: Involvement is the cause of pay.\nStep 2: The government will pay Unisys because of its involvement in the defense-procurement scandal.\nStep 3: Therefore, there is a causal relationship between involvement and pay.

####
Text: That would be a huge problem if it were to continue , because Unisys is betting its business on the assumption that customers want to move away from using operating systems that run on only one manufacturer 's equipment and toward systems -- mainly Unix -- that work on almost anyone 's machines . 
Source: betting
Target: problem
Question: Is there a causal relationship between betting  and problem, Let's think step by step?
Answer:Step 1: The text states that \"Unisys is betting its business on the assumption that customers want to move away from using operating systems that run on only one manufacturer 's equipment and toward systems -- mainly Unix -- that work on almost anyone 's machines\".\n\nStep 2: The text also states that \"That would be a huge problem if it were to continue\".\n\nStep 3: So, if Unisys's business is betting on the assumption that customers want to move away from using operating systems that run on only one manufacturer 's equipment and toward systems -- mainly Unix -- that work on almost anyone 's machines, and if that assumption is wrong, then Unisys's business will be in trouble.\n\nStep 4: So, there is a causal relationship between betting and problem.

"""
#  Example come from: "item_id": 11, CTB


zeroshot_cot_prompt = \
    """Answer the question by thinking step by step.
    """
def self_ask(source, target):
    prompt = f'''Question:Is there a causal relationship between {source} and {target}?
 - **Are follow up questions needed here: Yes.**
    - Follow up:What is {source}?
    - Intermediate answer:...

    - Follow up:What is {target}?
    - Intermediate answer:...

    - So the final answer is:...
    '''
    return prompt

one_shot_self_ask_prompt = f'''
Please answer the new questions given to you based on how you answered this example.
For example:
\"\"\"
Text: The projection sent Anheuser shares plunging $ 4.375 in New York Stock Exchange composite trading yesterday.
Source: projection
Target: plunging
Question: Is there a causal relationship between projection and plunging?
Answer:
    - Question 1:What is projection?
    - Answer:Projection is the act of predicting something.

    - Question 2:What is plunging?
    - Answer:Plunge is the act of falling suddenly and steeply.

    - Question 3:What about in this case?
    - Answer:In this case, the projection of Anheuser shares sent the shares plunging. This means that the prediction of Anheuser shares falling caused the shares to actually fall.

    - So the final answer is:Yes, there is a causal relationship between projection and plunging.
\"\"\"

'''

formal = '''Please you JUST answer me like {{\"Answer\":\"Yes\"}} or {{\"Answer\":\"No\"}}, in the format of JSON {{\"Answer\":\" \"}}.'''


def least_to_most_decomposer(source, target, text):
    least_to_most_decomposer_prompt = f"""
- In order to determine whether there is a causal relationship between {source} and {target}, you need to provide a series of sub-questions that lead us to the final answer. 
Follow the outline below to solve the question: 
    - **Question:**
        Is there a causal relationship between {source} and {target} in {text}?
    - **Answer:**    
        (To judge the causality of the statement "Is there a causal relationship between {source} and {target}?" we would need to know the answers to the following sub-questions.)

        - Subquestion 1: ...
        - Subquestion 2: ...
             
"""
    return least_to_most_decomposer_prompt

def least_to_most_subq_solver(source, target):
    least_to_most_subq_solver_prompt = f"""
Answer the sub-questions one by one according to the context. 

Q:Is there a causal relationship between {source} and {target}?
A:   
        - Subquestion 1: ...
        - Answer to subquestion 1: ...
        - Subquestion 2: ...
        - Answer to subquestion 2: ...

"""
    return least_to_most_subq_solver_prompt


CaCo_CoT_reasoner_prompt = """
Follow the outline below to solve the question: 
    - **Explanation of Terms**: 
    (You need to explain each term used in the question to remove ambiguity. )
        - Term 1: ...
        - Term 2: ...

    - **Subquestion Decomposition and Answering**: 
    (You need to decompose the question into several subquestions connected logically to arrive at the final answer. For each subquestion, provide your answer below it. )
        - Subquestion 1: ...
        - Answer to subquestion 1: ...
        - Subquestion 2: ...
        - Answer to subquestion 2: ...

    - **Rationale for Arriving at the Answer**: 
    (You need to reason step by step for the correct answer based on the previous information. )

    - **Provide Your Answer WITH TAGS**: 
    (You MUST choose ONE option, with the capital letter included in TAGS, e.g., <Answer>Yes/No</Answer>. If the answer cannot be determined, make an educated guess. )

"""

CaCo_CoT_reviewer_prompt = \
    f"""
You are an objective and fair reviewer. You are responsible for evaluating a possible solution following the outline: 
    - **Review Statements One by One**
        (Examine the following aspects in the solution. Especially, watch out for factualness errors and inference errors in the solution)
        - Evaluation on Explanation of Terms: 
        - Evaluation on Subquestion Decomposition and Answering: 
        - Evaluation on the Reasoning process: 
        - Evaluation on the Answer: 

    - **Reconsider the Question Step by Step and Consider Counterfactuals**
        - You need to perform counterfactuals to your results: 
            - Opposite answer (If your previous answer is that there is a causal relationship, the opposite answer is that there is no causal relationship. And vice versa.): 
            - What if we apply the opposite result: 
            - Will there be an contradiction: 

        - Your step-by-step reasoning to arrive at the most likely answer: 
        (Your reasoning MUST lead to a certain answer. )

    - **Provide Your Answer With TAGS** 
      (You MUST enclose your answer (MUST be certain) with TAGS, e.g., <Answer>Yes/No</Answer>. Make an educated guess if the answer CANNOT BE DETERMINED. )

"""
CaCo_CoT_formal = '''Please you JUST answer me like <Answer>Yes</Answer> or <Answer>No</Answer>.'''


