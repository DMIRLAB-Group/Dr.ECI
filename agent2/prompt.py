mediator_system = "You are an expert in Entity Extraction. You are good at extracting entities associated with source event and target event from the context text given to you. Note that the associated entities are at word level."
def mediator(source, target, text):
    prompt =f'''
 "text": {text}
"source event": {source},
"target event": {target},   
Question: Please give the entity that the source event and target event are related to.
###
Please follow the following process to answer the  Question:

        - Subquestion 1:Please search for all events or entities associated with the event "{source}" from "{text}", except the event "{source}" itself.
        - Answer to subquestion 1:(answer me at word level, such as "stay", "begged", "locked up","John","person".)

        - Subquestion 2:Please search for all events or entities associated with the event "{target}" from "{text}", except the event "{target}" itself.
        - Answer to subquestion 2:(answer me at word level, such as "stay", "begged", "locked up","John","person".)

        - Subquestion 3:Please find all events or entities in '{text}' that are associated with event '{source}' and event '{target}', except event '{source}' and event '{target}' themselves.
        - Answer to subquestion 3:(answer me at word level, such as "stay", "begged", "locked up","John","person".)

        - Subquestion 4: According to the previous Subquestion 3, if there are entities associated with event '{source}' and event '{target}', except event '{source}' and event '{target}' themselves. Give these entities in json format, such as {{["Entity 1", "Entity 2",...]}}. If they do not exist, answer directly {{"No"}}.
        - Answer to subquestion 4:
    '''
    return prompt

mediator_formal = "If there are related events or entities, please save them directly in JSON format, such as {{[\"entity 1\",\"entity 2\", ...]}}. If not, just answer No directly, such as {{\"Answer\":\"None\"}}."
