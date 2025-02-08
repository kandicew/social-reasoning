import obj.conversation as C
import pickle

# visualization and clustering of the reasons

all_path = '/PATH/data/all_obj.pkl'
with open(all_path, 'rb') as f:
    obj_lst = pickle.load(f)

def get_reasons(conv, aspect_idx):
    # 0: style, 1: content, 2: coordination, 3: engagement
    aspect = conv.get_aspects()[aspect_idx]
    a = aspect.reason_a
    b = aspect.reason_b
    reasons = []
    for i in a:
        reasons.append(i.text)
    for i in b:
        reasons.append(i.text)
    return reasons



from bertopic import BERTopic

from bertopic.representation import KeyBERTInspired


reasons = []
for asp in range(4): 
    for i in range(151):
        rs = get_reasons(obj_lst[i], asp)
        reasons += rs


classes = ['style']*906 +['content']*906 +['coordination']*906 +['engagement']*906
# Fine-tune your topic representations
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model)



topics, probs = topic_model.fit_transform(reasons)
topics_per_class = topic_model.topics_per_class(reasons, classes=classes)
topic_model.visualize_topics_per_class(topics_per_class)
