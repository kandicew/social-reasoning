from typing import List
import wrapper as W
class Reason:
    id: str
    text: str
    valid_potential: float
    grounding_potential: float
    
    def __init__(self, id, text, valid_potential, grounding_potential):
        self.id = id
        self.text = text
        self.valid_potential = valid_potential
        self.grounding_potential = grounding_potential

    def __str__(self) -> str:
        return f"Reason: {self.id}\n{self.text}\nValid: {self.valid_potential:.2f}\nGrouding: {self.grounding_potential:.2f}"

# rebuttal counters the reason with the same id
class Rebuttal:
    id: str
    text: str
    valid_potential: float
    rebuttal_potential: float

    def __init__(self, id, text, valid_potential, rebuttal_potential):
        self.id = id
        self.text = text
        self.valid_potential = valid_potential
        self.rebuttal_potential = rebuttal_potential
    
    def __str__(self) -> str:
        return f"Rebuttal: {self.id}\n{self.text}\nValid: {self.valid_potential:.2f}\nGrouding: {self.rebuttal_potential:.2f}"


class Aspect:
    name: str
    summary: str
    reason_a: List[Reason]
    reason_b: List[Reason]
    rebuttal_a: List[Rebuttal]
    rebuttal_b: List[Rebuttal]
    similarity_score: List[List]
    contradict_score_a: List[List]
    contradict_score_b: List[List]

    def __init__(self, name, summary, reason_a, reason_b, rebuttal_a, rebuttal_b, similarity_score, contradict_score_a, contradict_score_b):
        self.name = name
        self.summary = summary
        self.reason_a = reason_a
        self.reason_b = reason_b
        self.rebuttal_a = rebuttal_a
        self.rebuttal_b = rebuttal_b
        if len(similarity_score) == 0 and len(contradict_score_a) == 0 and len(contradict_score_b) == 0:
            reason_a_text = []
            reason_b_text = []
            for r in self.reason_a:
                reason_a_text.append(r.text)
                print(r.text)
            for r in self.reason_b:
                reason_b_text.append(r.text)
                print(r.text)
            
            _, self.similarity_score = W.generate_similar_likert(reason_a_text, reason_b_text)
            _,_,self.contradict_score_a, self.contradict_score_b = W.generate_contradict_likert(reason_a_text, reason_b_text)
        else:
            self.similarity_score = similarity_score
            self.contradict_score_a = contradict_score_a
            self.contradict_score_b = contradict_score_b

    
    def __str__(self) -> str:
        return f"Aspect: {self.name}\n\nSummary: {self.summary}\n\nReason_a:\n{self.reason_a[0].text}\n{self.reason_a[1].text}\n{self.reason_a[2].text}\n\nReason_b:\n{self.reason_b[0].text}\n{self.reason_b[1].text}\n{self.reason_b[2].text}"
    
    def get_reason_a_potentials(self):
        # return [[valid, grounding], [valid, grounding], [valid, grounding]]
        return [[self.reason_a[0].valid_potential, self.reason_a[0].grounding_potential],
                [self.reason_a[1].valid_potential, self.reason_a[1].grounding_potential],
                [self.reason_a[2].valid_potential, self.reason_a[2].grounding_potential]]
    
    def get_reason_b_potentials(self):
        # return [[valid, grounding], [valid, grounding], [valid, grounding]]
        return [[self.reason_b[0].valid_potential, self.reason_b[0].grounding_potential],
                [self.reason_b[1].valid_potential, self.reason_b[1].grounding_potential],
                [self.reason_b[2].valid_potential, self.reason_b[2].grounding_potential]]
    
    def get_rebuttal_a_potentials(self):
        # return [[valid, rebuttal], [valid, rebuttal], [valid, rebuttal]]
        return [[self.rebuttal_a[0].valid_potential, self.rebuttal_a[0].rebuttal_potential],
                [self.rebuttal_a[1].valid_potential, self.rebuttal_a[1].rebuttal_potential],
                [self.rebuttal_a[2].valid_potential, self.rebuttal_a[2].rebuttal_potential]]
    
    def get_rebuttal_b_potentials(self):
        # return [[valid, rebuttal], [valid, rebuttal], [valid, rebuttal]]
        return [[self.rebuttal_b[0].valid_potential, self.rebuttal_b[0].rebuttal_potential],
                [self.rebuttal_b[1].valid_potential, self.rebuttal_b[1].rebuttal_potential],
                [self.rebuttal_b[2].valid_potential, self.rebuttal_b[2].rebuttal_potential]]
    
    def print_info(self):
        print(self.name)
        print(self.summary)
        for i,j in zip(self.reason_a, self.rebuttal_a):
            print(i.__str__())
            print(j.__str__())
        for i,j in zip(self.reason_b, self.rebuttal_b):
            print(i.__str__())
            print(j.__str__())
        return


        

class Conversation:
    id: str
    text: str
    power: str
    style: Aspect
    content: Aspect
    coordination: Aspect
    engagement: Aspect

    def __init__(self, id, text, power, style, content, coordination, engagement):
        self.id = id
        self.text = text
        self.power = power
        self.style = style
        self.content = content
        self.coordination = coordination
        self.engagement = engagement

    def __str__(self) -> str:
        return f"Conversation: {self.id}\nPower: {self.power}"
    
    def get_aspects(self):
        return [self.style, self.content, self.coordination, self.engagement]


