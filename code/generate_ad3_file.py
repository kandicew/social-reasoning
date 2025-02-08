# input conversation object, generate corresponding ad3 file
from obj.conversation import Conversation, Aspect, Reason, Rebuttal
import calculate_potential as CP
import numpy as np

with open('/PATH/rules/rules.txt', 'r') as file:
    RULE_STR = file.read()

def gen_ad3_file(conv_obj: Conversation, file_path: str,
                 reason_vg=[1,1], reason_aspect=[1,1,1,1], rs=1,
                 rebuttal_vr=[1,1], rebuttal_aspect=[1,1,1,1], rbs=1,
                 similar=1, similar_top=1,
                 contradict=1, contradict_top=1,
                 logit=False):
    str_input='98\n75\n0\n0\n'
    reasons_potential = CP.get_reason_potentials(conv_obj=conv_obj, vg_scale=reason_vg, aspect_scale=reason_aspect)
    reason_a_potential = reasons_potential[0] # 4x3
    reason_b_potential = reasons_potential[1] # 4x3

    rebuttals_potentials = CP.get_rebuttal_potentials(conv_obj=conv_obj, vr_scale=rebuttal_vr, aspect_scale=rebuttal_aspect)
    rebuttal_a_potential = rebuttals_potentials[0] # 4x3
    rebuttal_b_potential = rebuttals_potentials[1] # 4x3

    similar_potentials = CP.get_similar_potentials(conv_obj=conv_obj, similar_scale=similar, top_num=similar_top) # 4x6
    contradict_potentials = CP.get_contradict_potentials(conv_obj=conv_obj, contradict_scale=contradict, top_num=contradict_top) # 4x2x3

    for i,j,k,l,m,n in zip(reason_a_potential, reason_b_potential, rebuttal_a_potential, rebuttal_b_potential, similar_potentials, contradict_potentials):
        # write reason a potential
        for p in i:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rs*logit_func(num))}\n'
            else:
                str_input += f'{str(rs*num)}\n'
        # write reason b potential
        for p in j:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rs*logit_func(num))}\n'
            else:
                str_input += f'{str(rs*num)}\n'
        # write rebuttal a potential
        for p in k:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rbs*logit_func(num))}\n'
            else:
                str_input += f'{str(rbs*num)}\n'
        # write rebuttal b potential
        for p in l:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rbs*logit_func(num))}\n'
            else:
                str_input += f'{str(rbs*num)}\n'
        # write similar potential
        for p in m:
            num = p/5
            if logit:
                str_input += f'{str(logit_func(num))}\n'
            else:
                str_input += f'{str(num)}\n'
        # write contradict potential
        for p_lst in n:
            for p in p_lst:
                num = p/5
                if logit:
                    str_input += f'{str(logit_func(num))}\n'
                else:
                    str_input += f'{str(num)}\n'
    # add rules lines
    str_input += RULE_STR
    # write to ad3 file
    with open(file_path, 'w') as f:
        f.write(str_input)
    return str_input


def gen_ad3_file_reason_only(conv_obj: Conversation, file_path: str,
                 reason_vg=[1,1], reason_aspect=[1,1,1,1],
                 logit=False):
    
    str_input='26\n3\n0\n0\n'
    reasons_potential = CP.get_reason_potentials(conv_obj=conv_obj, vg_scale=reason_vg, aspect_scale=reason_aspect)
    reason_a_potential = reasons_potential[0] # 4x3
    reason_b_potential = reasons_potential[1] # 4x3

    for i,j in zip(reason_a_potential, reason_b_potential):
        # write reason a potential
        for p in i:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(logit_func(num))}\n'
            else:
                str_input += f'{str(num)}\n'
        # write reason b potential
        for p in j:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(logit_func(num))}\n'
            else:
                str_input += f'{str(num)}\n'
        
    # add rules lines
    with open('/PATH/rules/rules_reasons.txt', 'r') as file:
        rules = file.read()
    str_input += rules
    # write to ad3 file
    with open(file_path, 'w') as f:
        f.write(str_input)
    return str_input

def gen_ad3_file_rebuttal(conv_obj: Conversation, file_path: str,
                 reason_vg=[1,1], reason_aspect=[1,1,1,1], rs=1,
                 rebuttal_vr=[1,1], rebuttal_aspect=[1,1,1,1], rbs=1,
                 logit=False):
    str_input='50\n27\n0\n0\n'
    reasons_potential = CP.get_reason_potentials(conv_obj=conv_obj, vg_scale=reason_vg, aspect_scale=reason_aspect)
    reason_a_potential = reasons_potential[0] # 4x3
    reason_b_potential = reasons_potential[1] # 4x3

    rebuttals_potentials = CP.get_rebuttal_potentials(conv_obj=conv_obj, vr_scale=rebuttal_vr, aspect_scale=rebuttal_aspect)
    rebuttal_a_potential = rebuttals_potentials[0] # 4x3
    rebuttal_b_potential = rebuttals_potentials[1] # 4x3


    for i,j,k,l in zip(reason_a_potential, reason_b_potential, rebuttal_a_potential, rebuttal_b_potential):
        # write reason a potential
        for p in i:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rs*logit_func(num))}\n'
            else:
                str_input += f'{str(rs*num)}\n'
        # write reason b potential
        for p in j:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rs*logit_func(num))}\n'
            else:
                str_input += f'{str(rs*num)}\n'
        # write rebuttal a potential
        for p in k:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rbs*logit_func(num))}\n'
            else:
                str_input += f'{str(rbs*num)}\n'
        # write rebuttal b potential
        for p in l:
            num = np.round(np.exp(p),2)
            if logit:
                str_input += f'{str(rbs*logit_func(num))}\n'
            else:
                str_input += f'{str(rbs*num)}\n'

    # add rules lines
    with open('/PATH/rules/rules_rr.txt', 'r') as file:
        rules = file.read()
    str_input += rules
    # write to ad3 file
    with open(file_path, 'w') as f:
        f.write(str_input)
    return str_input

# logit function with hyperparameter and normalization to none 0 value
def logit_func(p, k=5):
    new_p = np.exp(k*(p-1)) - 0.0067
    logit_p = np.log(new_p/(1-new_p))
    return logit_p+2.50667



'''
# code for generating the files
from tqdm import tqdm
for i in tqdm(range(151)):
    try:
        conv_obj = obj_lst[i]
        print(gen_ad3_file(conv_obj, f'/PATH/files/conv{i}.fg', rs=20, rbs=18))
        print(gen_ad3_file(conv_obj, f'/PATH//files/conv{i}_l.fg', logit=True, rs=20, rbs=18))
    except:
        pass

'''

