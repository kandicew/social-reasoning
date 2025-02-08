import numpy as np
# a function to generate different potential for the similar variable based on the llm generated likert answer
# give a list of list of score with potention (a list contains 6 list, where in each list, there are 5 [score, logprob]),
# top_num is how many probabilities we want to take into consideration
# if top_num is 1, we are only using the top generated score as potential
# if top_num is 2, we are using the top 2 (sum of score * weight in percentile)
def similar_potential(lst, top_num=1):
    weight_lst = []
    for l in lst:
        # need to make sure the most likely generated token is always a number
        if top_num == 1:
            weight_lst.append(int(l[0][0].strip()))
        else:
            sub_lst = l[:top_num]
            total = 0
            for s in sub_lst:
                weight = np.round(np.exp(s[1]),2)
                try:
                    total += int(s[0].strip()) * weight
                except:
                    pass
            weight_lst.append(total)
    return weight_lst


#########################################using conversation objects to generate potentials#########################################
from obj.conversation import Conversation, Aspect, Reason, Rebuttal
from typing import List
import numpy as np
def get_reason_potentials(conv_obj: Conversation, vg_scale=[1,1], aspect_scale=[1,1,1,1]) -> List[List]:
    # input a conversation object with 
    # optional validation and grounding scales,
    # the hyperparameters of validation and grounding for all aspects remain the same
    # optional aspect scales
    # each aspect could weight differently
    # return a list of lists (4*3), corresponding to the potential of each reasons grouped by aspects
    aspects = conv_obj.get_aspects()
    reason_a_potentials = []
    reason_b_potentials = []
    for a in aspects:
        potentials = a.get_reason_a_potentials()
        reasons_a_po = np.array(potentials) @ np.transpose(np.array(vg_scale))
        reason_a_potentials.append(list(reasons_a_po))
        potentials = a.get_reason_b_potentials()
        reasons_b_po = np.array(potentials) @ np.transpose(np.array(vg_scale))
        reason_b_potentials.append(list(reasons_b_po))
    
    reason_a_potentials_scaled = []
    reason_b_potentials_scaled = []
    for scale,ra,rb in zip(aspect_scale, reason_a_potentials, reason_b_potentials):
        temp_a = []
        temp_b = []
        for p,q in zip(ra, rb):
            temp_a.append(scale*p)
            temp_b.append(scale*q)
        reason_a_potentials_scaled.append(temp_a)
        reason_b_potentials_scaled.append(temp_b)
    return [reason_a_potentials_scaled, reason_b_potentials_scaled]


def get_rebuttal_potentials(conv_obj: Conversation, vr_scale=[1,1], aspect_scale=[1,1,1,1]) -> List[List]:
    # input a conversation object with 
    # optional validation and grounding scales,
    # the hyperparameters of validation and grounding for all aspects remain the same
    # optional aspect scales
    # each aspect could weight differently
    # return a list of lists (4*3), corresponding to the potential of each reasons grouped by aspects
    aspects = conv_obj.get_aspects()
    reason_a_potentials = []
    reason_b_potentials = []
    for a in aspects:
        potentials = a.get_rebuttal_a_potentials()
        reasons_a_po = np.array(potentials) @ np.transpose(np.array(vr_scale))
        reason_a_potentials.append(list(reasons_a_po))
        potentials = a.get_rebuttal_b_potentials()
        reasons_b_po = np.array(potentials) @ np.transpose(np.array(vr_scale))
        reason_b_potentials.append(list(reasons_b_po))
    
    reason_a_potentials_scaled = []
    reason_b_potentials_scaled = []
    for scale,ra,rb in zip(aspect_scale, reason_a_potentials, reason_b_potentials):
        temp_a = []
        temp_b = []
        for p,q in zip(ra, rb):
            temp_a.append(scale*p)
            temp_b.append(scale*q)
        reason_a_potentials_scaled.append(temp_a)
        reason_b_potentials_scaled.append(temp_b)
    return [reason_a_potentials_scaled, reason_b_potentials_scaled]


def get_similar_potentials(conv_obj, similar_scale=1, top_num=1) ->List[List]:
    # input a conversation object with 
    # optional scale to weight similarity scores
    # optional top_num input to calculate the top returned tokens
    # return a list of lists (4*6), corresponding to the potential each similar pair of reasons grouped by aspects
    scores = []
    aspects = conv_obj.get_aspects()
    for a in aspects:
        ss = similar_potential(a.similarity_score, top_num=top_num)
        scores.append([similar_scale*i for i in ss])
    return scores


def get_contradict_potentials(conv_obj, contradict_scale=1, top_num=1) ->List[List]:
    # input a conversation object with 
    # optional scale to weight contradict scores
    # optional top_num input to calculate the top returned tokens
    # return a list of lists (4*2*3)
    # [style, content, coordination, engagenemt]
    # for each of the aspect [contradition scores for reasons a, contradiction scores for reasons b]
    # for each contradict scores [a1a2, a1a3, a2a3]
    scores = []
    aspects = conv_obj.get_aspects()
    for a in aspects:
        cs_a = similar_potential(a.contradict_score_a, top_num=top_num)
        cs_b = similar_potential(a.contradict_score_b, top_num=top_num)
        scores.append([[contradict_scale*i for i in cs_a], [contradict_scale*i for i in cs_b]])
    return scores

