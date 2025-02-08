import data_prep as DP
import generate as G
import prompts as P


def load_data():
    convs, labels = DP.get_conv()
    return convs, labels

def generate_summary(aspect, conversation):
    prompt = P.create_summary_prompt(aspect, conversation)
    res = G.chat_gpt(prompt)
    start_str = aspect + ' of A:'
    if not res.startswith(start_str):
        res = start_str + '\n' + res
    return res

def generate_reasons(summary):
    prompt_a = P.create_reason_prompt('A', summary)
    prompt_b = P.create_reason_prompt('B', summary)
    reasons_a = G.chat_gpt(prompt_a)
    reasons_b = G.chat_gpt(prompt_b)
    lst = [reasons_a, reasons_b]
    return lst

'''
def generate_reasons_s(aspect, summary):
    def split_sum(a,summ):
        s_summ = summ.split(':')
        a = s_summ[1].split(f'{a} of ')[0].strip()
        return a, s_summ[-1].strip()
    sum_a, sum_b = split_sum(aspect, summary)
    prompt_a = P.create_reason_prompt_s('A', sum_a)
    prompt_b = P.create_reason_prompt_s('B', sum_b)
    reasons_a = G.chat_gpt(prompt_a)
    reasons_b = G.chat_gpt(prompt_b)
    lst = [reasons_a, reasons_b]
    return lst
'''

def process_reasons(reasons):
    '''
    gvien [all reasons for a in a string, all reasons for b in a string]
    process the output
    return [[list of reasons for a], [list of reasons for b]]
    '''
    lst = []
    for i in reasons:
        sub_lst = []
        rs = i.splitlines()
        for r in rs:
            r = r[:-1]
            if r.startswith('-'):
                r = r[3:]
            sub_lst.append(r)
        lst.append(sub_lst)
    return lst



def generate_rebuttal(conversation, reasons):
    '''
    given output from process_reasons, generate the rebuttals for each of the reasons
    return the rebuttals in the same format [[list of rebuttals against a], [list of rebuttals against b]]
    '''
    reason_lst = process_reasons(reasons)
    reasons_a = reason_lst[0]
    reasons_b = reason_lst[1]
    rebuttals_b = []
    rebuttals_a = []
    for i in reasons_a:
        prompt = P.create_rebuttal_prompt(conversation=conversation, high='A', reason=i)
        res = G.chat_gpt(prompt)
        rebuttals_b.append(res)
    for i in reasons_b:
        prompt = P.create_rebuttal_prompt(conversation=conversation, high='B', reason=i)
        res = G.chat_gpt(prompt)
        rebuttals_a.append(res)
    return [reasons_a, reasons_b, rebuttals_b, rebuttals_a]
    
def generate_valid_with_prob(reasons):
    reason_lst = process_reasons(reasons)
    reasons_a = reason_lst[0]
    reasons_b = reason_lst[1]
    lst_res_a = []
    lst_res_b = []
    lst_prob_a = []
    lst_prob_b = []
    for i in reasons_a:
        prompt = P.create_validation_prompt(high='A', reason=i)
        res, prob = G.gen_score_with_prob(prompt)
        lst_res_a.append(res)
        lst_prob_a.append(prob)
    for i in reasons_b:
        prompt = P.create_validation_prompt(high='B', reason=i)
        res, prob = G.gen_score_with_prob(prompt)
        lst_res_b.append(res)
        lst_prob_b.append(prob)
    
    return lst_res_a, lst_res_b, lst_prob_a, lst_prob_b

# give a list of reasons and a list of rebuttals,
# return the potential scores [[[response_valid, score_valid, respond_ground, score_ground] for reasons_a], [[response and scores] for rebuttals_b], [[response and scores] for reasons_b], [[response and scores] for rebuttals_a]]
def generate_valid_ground_with_prob_rr(conversation, reasons_a, rebuttals_b, reasons_b, rebuttals_a):
    lst_prob_reason_a = []
    lst_prob_rebuttal_b = []
    lst_prob_reason_b = []
    lst_prob_rebuttal_a = []
    for i in reasons_a:
        prompt = P.create_validation_prompt(high='A', reason=i)
        res, prob = G.gen_score_with_prob(prompt)
        prompt_g = P.create_grounding_prompt(conversation=conversation, reason=i, high='A')
        resg, probg = G.gen_score_with_prob(prompt_g)
        lst_prob_reason_a.append([prob[0], prob[1], probg[0], probg[1]])
    for i,j in zip(reasons_a, rebuttals_b):
        prompt = P.create_validation_prompt_rebuttal(high='B', rebuttal=i, reason=j)
        res, prob = G.gen_score_with_prob(prompt)
        prompt_g = P.create_grounding_rebuttal_prompt(conversation=conversation, reason=i)
        resg, probg = G.gen_score_with_prob(prompt_g)
        lst_prob_rebuttal_b.append([prob[0], prob[1], probg[0], probg[1]])
    for i in reasons_b:
        prompt = P.create_validation_prompt(high='B', reason=i)
        res, prob = G.gen_score_with_prob(prompt)
        prompt_g = P.create_grounding_prompt(conversation=conversation, reason=i, high='B')
        resg, probg = G.gen_score_with_prob(prompt_g)
        lst_prob_reason_b.append([prob[0], prob[1], probg[0], probg[1]])
    for i,j in zip(rebuttals_a, reasons_b):
        prompt = P.create_validation_prompt_rebuttal(high='A', rebuttal=i, reason=j)
        res, prob = G.gen_score_with_prob(prompt)
        prompt_g = P.create_grounding_rebuttal_prompt(conversation=conversation, reason=i)
        resg, probg = G.gen_score_with_prob(prompt_g)
        lst_prob_rebuttal_a.append([prob[0], prob[1], probg[0], probg[1]])
    
    return lst_prob_reason_a, lst_prob_rebuttal_b, lst_prob_reason_b, lst_prob_rebuttal_a


import numpy as np
def generate_similar(reasons):
    '''
    give [all reason of a as a string, all reason for b as a string]
    return 3*3 matrix, m[i][j] is the similarity score of a[i] and b[j], j always >= i
    (only the lower triangle of the matrix is modified)
    '''
    reason_lst = process_reasons(reasons)
    reasons_a = reason_lst[0]
    reasons_b = reason_lst[1]
    similarity_matrix = np.tri(N=3, dtype=int)
    i = 0
    while i < 3:
        j = i
        while j < 3:
            description_a = reasons_a[i]
            description_b = reasons_b[j]
            prompt = P.create_same_prompt(description_a=description_a, description_b=description_b)
            res = G.chat_gpt(prompt)
            print("score of " + description_a + " and " + description_b + "is: " + res)
            try:
                similarity_matrix[j][i] = int(res.strip())
            except:
                print(res)
            j += 1
        i += 1
    return similarity_matrix.tolist()



def generate_similar_likert(reasons_a, reasons_b):
    lst_res = []
    lst_dict = []
    i = 0
    while i < 3:
        j = i
        while j < 3:
            description_a = reasons_a[i]
            description_b = reasons_b[j]
            prompt = P.create_same_prompt_likert(description_a=description_a, description_b=description_b)
            res, dict = G.gen_likert_score_with_prob(prompt)
            lst_res.append(res)
            lst_dict.append(dict)
            '''
            print("score of " + description_a + " and " + description_b + "is: " + res)
            try:
                similarity_matrix[j][i] = int(res.strip())
            except:
                print(res)
            '''
            j += 1
        i += 1
    return lst_res, lst_dict

def generate_contradict_likert(reasons_a, reasons_b):

    lst_res_a = []
    lst_dict_a = []
    lst_res_b = []
    lst_dict_b = []

    #matrix_a = np.tri(N=3, k=-1, dtype=int)
    #matrix_b = np.tri(N=3, k=-1, dtype=int)
    # reasons for a
    # 1 & 2
    description_a = reasons_a[0]
    description_b = reasons_a[1]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_a.append(res)
    lst_dict_a.append(dict)
    '''
    try:
        matrix_a[1][0] = int(res.strip())
    except:
        print(res)
    '''
    
    # 1 & 3
    description_a = reasons_a[0]
    description_b = reasons_a[2]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_a.append(res)
    lst_dict_a.append(dict)
    
    # 2 & 3
    description_a = reasons_a[1]
    description_b = reasons_a[2]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_a.append(res)
    lst_dict_a.append(dict)
    

    # reasons for b
    # 1 & 2
    description_a = reasons_b[0]
    description_b = reasons_b[1]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_b.append(res)
    lst_dict_b.append(dict)
    
    # 1 & 3
    description_a = reasons_b[0]
    description_b = reasons_b[2]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_b.append(res)
    lst_dict_b.append(dict)
    
    # 2 & 3
    description_a = reasons_b[1]
    description_b = reasons_b[2]
    prompt = P.create_contradict_prompt_likert(description_a=description_a, description_b=description_b)
    res, dict = G.gen_likert_score_with_prob(prompt)
    #print("score of " + description_a + " and " + description_b + "is: " + res)
    lst_res_b.append(res)
    lst_dict_b.append(dict)
    
    return lst_res_a, lst_res_b, lst_dict_a, lst_dict_b


def generate_contradict(reasons):
    '''
    give [all reason of a as a string, all reason for b as a string]
    return 2 3*3 matrices, one for the contradiction scores of reasons for a,
     one for the contradiction scores of reasons for b.
    m[i][j] is the similarity/contradiction score of a[i] and a[j], j always >= i
    (only the lower triangle of the matrix is modified)
    '''
    reason_lst = process_reasons(reasons)
    reasons_a = reason_lst[0]
    reasons_b = reason_lst[1]
    matrix_a = np.tri(N=3, k=-1, dtype=int)
    matrix_b = np.tri(N=3, k=-1, dtype=int)
    # reasons for a
    # 1 & 2
    description_a = reasons_a[0]
    description_b = reasons_a[1]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_a[1][0] = int(res.strip())
    except:
        print(res)
    
    # 1 & 3
    description_a = reasons_a[0]
    description_b = reasons_a[2]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_a[2][0] = int(res.strip())
    except:
        print(res)
    
    # 2 & 3
    description_a = reasons_a[1]
    description_b = reasons_a[2]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_a[2][1] = int(res.strip())
    except:
        print(res)
    

    # reasons for b
    # 1 & 2
    description_a = reasons_b[0]
    description_b = reasons_b[1]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_b[1][0] = int(res.strip())
    except:
        print(res)
    
    # 1 & 3
    description_a = reasons_b[0]
    description_b = reasons_b[2]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_b[2][0] = int(res.strip())
    except:
        print(res)
    
    # 2 & 3
    description_a = reasons_b[1]
    description_b = reasons_b[2]
    prompt = P.create_contradict_prompt(description_a=description_a, description_b=description_b)
    res = G.chat_gpt(prompt)
    print("score of " + description_a + " and " + description_b + "is: " + res)
    try:
        matrix_b[2][1] = int(res.strip())
    except:
        print(res)
    
    return [matrix_a.tolist(), matrix_b.tolist()]


import pandas as pd
# generation raw summary for each aspect and raw reasons (list of two strings) given a conversation
def generate_pipe(conversation):
    aspects = ['style', 'content', 'coordination', 'engagement']
    summ_dict = {'style': '', 'content': '', 'coordination': '', 'engagement': ''}
    reasons_dict = {'style': [], 'content': [], 'coordination': [], 'engagement': []}
    for a in aspects:
        summ = generate_summary(a, conversation)
        summ_dict[a] = summ
        reasons = generate_reasons(summ)
        reasons_dict[a] = reasons
    return summ_dict, reasons_dict

