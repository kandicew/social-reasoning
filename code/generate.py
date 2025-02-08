from openai import OpenAI

client = OpenAI(api_key = OPENAI_KEY)

def chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    message = response.choices[0].message.content.strip()
    return message


import numpy as np
# give a prompt that ask for the similarity score based on the likert scale
# return the output and a list of the top 5 token with logprob [['token (should be a number in 1-5)', logprob (float)], []...]
def gen_likert_score_with_prob(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5
    )
    message = response.choices[0].message.content.strip()
    logprobs_gen = response.choices[0].logprobs.content[0].top_logprobs
    prob_lst = []
    for i, lp in enumerate(logprobs_gen, start=1):
        prob_lst.append([lp.token, lp.logprob])
        print(f'output {i}: {lp.token}\nlogprob: {lp.logprob}\nprobability: {np.round(np.exp(lp.logprob)*100,2)}')
    return message, prob_lst

# for y/n questions
def gen_score_with_prob(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        logprobs=True
    )
    message = response.choices[0].message.content.strip()
    logprobs_gen = response.choices[0].logprobs.content[0]
    #print(logprobs_gen)
    return message, [logprobs_gen.token, logprobs_gen.logprob]