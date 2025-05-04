reasoner_system = '''You are an expert at identifying causal relationships between events in text, both implicit and explicit. Please answer this question based on the context you are given. '''


def reasoner(source, target, text, mediator_both):
    prompt = f'''

###
Please follow the following rules to answer the Question:
    - **According to each given causal rule, determine whether there is a causal relationship between X({source}) and Y({target}):**
    (If Z exists, determine which of the following rules each Z can satisfy with events X and Y. If Z does not exist, directly analyze what rules are satisfied between X and Y.)

where:
    - TEXT: {text}
    - X: {source}
    - Y: {target}
    - Z: {mediator_both}

- Rules:
        - Rule 1: Explicit Causation Words
            */In the text, there are verbs or verb phrases that clearly express causality, and they relate X and Y. The relationship between them can be expressed as: X cause/lead/due to (Explicit causation words) Y. So there is a causal relationship between X and Y.Explicit causation words(In addition to X and Y themselves, verbs or verb phrases that can clearly express causal relationships.):
            "cause," "result," "lead," "create," "cause by," "lead to," "result in," "due to", "because of","by", etc.*/
        - Answer to Rule 1: (If this rule is met, please give {{\"Rule 1\":\"Explicit Causation Words\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 2: Implicit Causation Words
            */There are implicit clue words in the text that imply a causal relationship, and they associate X and Y. These implicit clue words are expressed in natural language as being likely to cause event X and event Y to have a causal relationship. The relationship between them can be expressed as: X aggravate/force/provoke (Implicit causation words) Y. So there is a causal relationship between X and Y.Implicit causation words(In addition to X and Y themselves, verbs or verb phrases that implicitly express causal relationships can be used.):
            "spark," "aggravate," "stir," "precipitate," "ignite," "pave," "exacerbate," "force ," "arouse," "worsen," "result from," "reduce," "provoke," "inflict," "fuel," "stem," "prompt," "alleviate," "trigger," "increase with ," "contribute to" etc.*/
        - Answer to Rule 2: (If this rule is met, please give {{\"Rule 2\":\"Implicit Causation Words\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 3: Causal Order( or common sense knowledge)
            */According to common sense knowledge, if event X occurs before event Y and can be expressed as: X causes Y, then there is a causal relationship between X and Y. If the expression "X causes Y" does not follow common sense or is incorrectly expressed, then there is no causal relationship between them. */
        - Answer to Rule 3: (If this rule is met, please give {{\"Rule 3\":\"Causal Order\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 4: The Coreference event of event X
            */In the text, if an event with the same or similar meaning as the X can be found, and this similar event has a causal relationship with the Y event, then there is also a causal relationship between the X and the Y.*/
        - Answer to Rule 4: (If this rule is met, please give {{\"Rule 4\":\"The Coreference event of event X\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 5: The Coreference event of event Y
        */In the text, if an event with the same or similar meaning as the Y can be found, and this similar event has a causal relationship with the X, then there is also a causal relationship between the X and the Y.*/
        - Answer to Rule 5: (If this rule is met, please give {{\"Rule 5\":\"The Coreference event of event Y\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 6: Chain
        */In "text", there is an entity or event(Z) that both events X and Y are related to.  First,it satisfies that it is related to X and Y respectively, then it satisfies that X acts on Z and Z acts on Y, and finally they form a causal chain: X->Z->Y. Therefore, it can be concluded that there is a causal relationship between X and Y. */
        - Answer to Rule 6: (If this rule is met, please give {{\"Rule 6\":\"Chain\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 7: Collider
        */In "text", there is an entity or event(Z) that both events X and Y are related to.  First, it satisfies that it is related to X and Y respectively, and then it satisfies that X acts on Z and Y acts on Z. Finally, X, Z and Y form the Collider structure. Therefore, it can be concluded that there is a causal relationship between X and Y.*/
        - Answer to Rule 7: (If this rule is met, please give {{\"Rule 7\":\"Collider\"}} in JSON format, otherwise answer {{\"No\"}}.

        - Rule 8: Fork
        */In "text", there is an entity or event(Z) that both events X and Y are related to. First, it satisfies that it is related to X and Y respectively, and then it satisfies that Z acts on X and Z acts on Y. Finally X, Z and Y form the Fork structure. Therefore, it can be concluded that there is a causal relationship between X and Y.*/
        - Answer to Rule 8: (If this rule is met, please give {{\"Rule 8\":\"Fork\"}} in JSON format, otherwise answer {{\"No\"}}.        
    
        - Give your analysis first, and finally give the answer in JSON format. 
 '''

    return prompt
