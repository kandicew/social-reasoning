import obj.conversation as OBJ

import data_prep as DP
import wrapper as W


def load_data():
    convs, labels = DP.get_conv()
    return convs, labels

convs, labels = load_data()

def gen_conversation_object(id, text, power):
    # generate summary and reasons from conversations:
    summ_dict, reasons_dict = W.generate_pipe(conversation=text)
    aspect_obj = []
    for asp in summ_dict:
        summary = summ_dict[asp]
        reasons = reasons_dict[asp]
        # give reasons, return [reasons_a, reasons_b, rebuttals_b, rebuttals_a]
        reasons_and_rebuttals = W.generate_rebuttal(text, reasons=reasons)
        reasons_a = reasons_and_rebuttals[0]
        reasons_b = reasons_and_rebuttals[1]
        rebuttals_b = reasons_and_rebuttals[2]
        rebuttals_a = reasons_and_rebuttals[3]
        lra, lrbb, lrb, lrba = W.generate_valid_ground_with_prob_rr(text, reasons_a, rebuttals_b, reasons_b, rebuttals_a)
        reason_a_obj = []
        rebuttal_b_obj = []
        reason_b_obj = []
        rebuttal_a_obj = []
        for i in range(3):
            ra = reasons_a[i]
            rb = reasons_b[i]
            rbb = rebuttals_b[i]
            rba = rebuttals_a[i]
            print(lra[i])
            vp, gp = potential_covert_helper(lra[i])
            robj = OBJ.Reason(id=f'{id}_{asp}_A_{i}', text=ra, valid_potential=vp, grounding_potential=gp)
            print(robj.__str__())
            reason_a_obj.append(robj)
            vp, gp = potential_covert_helper(lrb[i])
            robj = OBJ.Reason(id=f'{id}_{asp}_B_{i}', text=rb, valid_potential=vp, grounding_potential=gp)
            reason_b_obj.append(robj)
            print(robj.__str__())
            vp, gp = potential_covert_helper(lrbb[i])
            rebuttal_b_obj.append(OBJ.Rebuttal(id=f'{id}_{asp}_A_{i}', text=rbb, valid_potential=vp, rebuttal_potential=gp))
            vp, gp = potential_covert_helper(lrba[i])
            rebuttal_a_obj.append(OBJ.Rebuttal(id=f'{id}_{asp}_B_{i}', text=rba, valid_potential=vp, rebuttal_potential=gp))
        aobj = OBJ.Aspect(name=asp, summary=summary, reason_a=reason_a_obj, reason_b=reason_b_obj, rebuttal_a=rebuttal_b_obj, rebuttal_b=rebuttal_a_obj, similarity_score=[], contradict_score_a=[], contradict_score_b=[])
        print(aobj.__str__())
        aspect_obj.append(aobj)    

    conv_obj = OBJ.Conversation(id=id, text=text, power=power, style=aspect_obj[0], content=aspect_obj[1], coordination=aspect_obj[2], engagement=aspect_obj[3])
    return conv_obj

import numpy as np
def potential_covert_helper(lst):
    # input [yes/no, valid_potential, yes/no, grounding_potential]
    # output [valid_potential_for_yes, grounding_potential_for_yes]
    if 'yes' in lst[0].lower():
        vp = lst[1]
    else:
        vp = np.log(1-np.exp(lst[1]))
    if 'yes' in lst[2].lower():
        gp = lst[3]
    else:
        gp = np.log(1-np.exp(lst[3]))
    return vp, gp

from tqdm import tqdm
# return the generated conversation objs in a list
# could append to existing list and save to pickle file
def generate_conv_obj_batch(start_idx, end_idx, conv_lst, label_lst):
    lst = []
    for i in tqdm(range(start_idx, end_idx)):
        try:
            conv_obj = gen_conversation_object(str(i+1), conv_lst[i], label_lst[i])
            lst.append(conv_obj)
        except:
            lst.append('exception')
    return lst


def generate_conv_obj_lst(lst_idx, conv_lst, label_lst):
    lst = []
    for i in tqdm(lst_idx):
        try:
            conv_obj = gen_conversation_object(str(i+1), conv_lst[i], label_lst[i])
            lst.append(conv_obj)
        except:
            lst.append('exception')
    return lst

'''
import pickle
all_path = '/PATH/data/all_obj.pkl'
with open(all_path, 'rb') as f:
    obj_lst = pickle.load(f)


example = obj_lst[20]
e_style = example.style
print(example.text)
print('------------------------')
print(e_style.summary)

a = e_style .reason_a
r = e_style .rebuttal_a
b = e_style .reason_b
rb = e_style.rebuttal_b

for rea, reb in zip(a,r):
    #print(rea.text)
    #print(reb.text)
    #print('------------------------')
    break

#print(b[0].text)
print(rb[0].text)
'''