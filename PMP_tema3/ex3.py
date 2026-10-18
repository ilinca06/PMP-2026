import random
import math
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# a) Simulare 10000 de jocuri
def simuleaza():
    incepe = random.choice([0, 1]) # 0 = J0, 1 = J1
    n = random.randint(1, 6)
    aruncari = 2 * n
    
    if incepe == 0:
        # J1 arunca moneda masluita (p=4/7)
        m = sum(1 for _ in range(aruncari) if random.random() < 4/7)
        return 'J0' if n >= m else 'J1'
    else:
        # J0 arunca moneda normala (p=0.5)
        m = sum(1 for _ in range(aruncari) if random.random() < 0.5)
        return 'J1' if n >= m else 'J0'

castig_j0 = sum(1 for _ in range(10000) if simuleaza() == 'J0')
print("a) Rezultate simulare (10000 jocuri):")
print(f"J0 a castigat: {castig_j0} | J1 a castigat: {10000 - castig_j0}\n")


# b) Reteaua Bayesiana 
model = DiscreteBayesianNetwork([('S', 'M'), ('N', 'M')])

# S = Cine incepe (50% - 50%), N = Zarul (1/6 pentru fiecare fata)
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.5], [0.5]])
cpd_n = TabularCPD(variable='N', variable_card=6, values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]])

def prob_binomiala(k, n_total, p):
    if k > n_total: return 0.0
    return math.comb(n_total, k) * (p**k) * ((1-p)**(n_total-k))

# Construim matricea M (Steme)
val_m = []
for m in range(13): # de la 0 la 12 steme maxime
    rand = []
    # S=0 (Joaca J1 cu p=4/7)
    for n in range(6): rand.append(prob_binomiala(m, 2*(n+1), 4/7))
    # S=1 (Joaca J0 cu p=0.5)
    for n in range(6): rand.append(prob_binomiala(m, 2*(n+1), 0.5))
    val_m.append(rand)

cpd_m = TabularCPD(variable='M', variable_card=13, values=val_m, evidence=['S', 'N'], evidence_card=[2, 6])

model.add_cpds(cpd_s, cpd_n, cpd_m)
model.check_model()


# c) Inferenta: Cine a inceput daca a picat fix 1 stema (M=1)
infer = VariableElimination(model)
rez_c = infer.query(['S'], evidence={'M': 1})
print("c) Cine a inceput (S) daca s-a obtinut 1 stema (M=1):")
print(rez_c)