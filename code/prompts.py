SUMMARY_TEMPLATE = '/PATH/template/summary_template.txt'
REASON_TEMPLATE = '/PATH/template/reason_template.txt'
GROUNDING_TEMPLATE = '/PATH/template/grounding_template.txt'
GROUNDING_REBUTTAL_TEMPLATE = '/PATH/template/grounding_rebuttal_template.txt'
VALIDATION_TEMPLATE = '/PATH/template/validation_template.txt'
VALIDATION_REBUTTAL_TEMPLATE = '/PATH/template/validation_rebuttal_template.txt'
REBUTTAL_TEMPLATE = '/PATH/template/rebuttal_template.txt'
SAME_TEMPLATE = '/PATH/template/same_template.txt'
CONTRADICT_TEMPLATE = '/PATH/template/contradict_template.txt'
SAME_LIKERT = '/PATH/template/same_likert.txt'
CONTRADICT_LIKERT = '/PATH/template/contradict_likert.txt'
TEMPLATE_DIR = '/PATH/template/'

definition = {'style': 'Style encompasses the tone, manner, and language used during the conversation. It can range from formal to informal, polite to blunt, friendly to hostile, etc.',
              'content': 'Content is the substance or subject matter of the conversation. It includes the topics being discussed, the information exchanged, and the sentence type used.',
              'coordination': 'Coordination is how participants manage turn-taking, interruptions, and transitions between topics. It involves maintaining a balance between speaking and listening, ensuring everyone has a chance to contribute.',
              'engagement': 'Engagement is the level of interest and involvement of participants in the conversation. Engaging conversations often involve asking questions, sharing personal experiences, and expressing empathy.'}


def create_summary_prompt(aspect, conversation, template=SUMMARY_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(aspect=aspect, aspect_definition=definition[aspect], conversation=conversation)
    return tem

def create_reason_prompt(high, summary, template=REASON_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(high=high, low="A" if high=="B" else "B", summary=summary)
    return tem


def create_grounding_prompt(high, conversation, reason, template=GROUNDING_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(high=high, low="A" if high=="B" else "B", conversation=conversation, reason=reason)
    return tem

def create_grounding_rebuttal_prompt(conversation, reason, template=GROUNDING_REBUTTAL_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(conversation=conversation, reason=reason)
    return tem

def create_validation_prompt(high, reason, template=VALIDATION_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(high=high, low="A" if high=="B" else "B", reason=reason)
    return tem

def create_rebuttal_prompt(high, conversation, reason, template=REBUTTAL_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(high=high, low="A" if high=="B" else "B", conversation=conversation, reason=reason)
    return tem

def create_validation_prompt_rebuttal(high, rebuttal, reason, template=VALIDATION_REBUTTAL_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(high=high, low="A" if high=="B" else "B", rebuttal=rebuttal, reason=reason)
    return tem

def create_same_prompt(description_a, description_b, template=SAME_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(description_a=description_a, description_b=description_b)
    return tem

def create_same_prompt_likert(description_a, description_b, template=SAME_LIKERT):
    with open(template) as f:
        t = f.read()
    tem = t.format(description_a=description_a, description_b=description_b)
    return tem

def create_contradict_prompt(description_a, description_b, template=CONTRADICT_TEMPLATE):
    with open(template) as f:
        t = f.read()
    tem = t.format(description_a=description_a, description_b=description_b, template=CONTRADICT_TEMPLATE)
    return tem

def create_contradict_prompt_likert(description_a, description_b, template=CONTRADICT_LIKERT):
    with open(template) as f:
        t = f.read()
    tem = t.format(description_a=description_a, description_b=description_b)
    return tem


