import numpy as np
from hmmlearn import hmm 

# stari: camerele 1, 2, 3, 4
camere=[1, 2, 3, 4]

# secventa red, green, blue, blue, red ([0, 1, 2, 2, 0])
obs=np.array([[0], [1], [2], [2], [0]])

#distributia uniforma
pi=np.array([0.25, 0.25, 0.25, 0.25])

#matricea de emisie B
B = np.array([
    [0.7, 0.2, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
    [0.5, 0.2, 0.3]
])

# sub a)
#matricea de tranzitie A
A_a = np.array([
    [0.2, 0.4, 0.0, 0.4],
    [0.4, 0.2, 0.4, 0.0],
    [0.0, 0.4, 0.2, 0.4],
    [0.4, 0.0, 0.4, 0.2]
])

model_a=hmm.CategoricalHMM(n_components=4)
model_a.startprob_ = pi
model_a.transmat_ = A_a
model_a.emissionprob_ = B

#algoritmul viterbi
log_p_a, path_a = model_a.decode(obs, algorithm="viterbi")
traseu_a=[camere[i] for i in path_a]
prob_a=np.exp(log_p_a)

print("cea mai probabila secventa de camere ", traseu_a)
print(f"probabilitatea comuna {prob_a:.9f}")

# sub b)
#matricea de tranzitie 
A_b = np.array([
    [0.0, 0.5, 0.0, 0.5],
    [0.5, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.5, 0.0, 0.5, 0.0]
])

model_b = hmm.CategoricalHMM(n_components=4)
model_b.startprob_ = pi
model_b.transmat_ = A_b
model_b.emissionprob_ = B

#algoritmul viterbi
log_p_b, path_b = model_b.decode(obs, algorithm="viterbi")
traseu_b = [camere[i] for i in path_b]
prob_b = np.exp(log_p_b)

print("\n cea mai probabila secventa de camere ", traseu_b)
print(f"probabilitatea comuna {prob_b:.9f}")

