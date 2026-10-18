from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# a) Definim modelul si adaugam probabilitatile
model = DiscreteBayesianNetwork([('B', 'T'), ('B', 'X'), ('B', 'D'), ('T', 'D')])

# Boala (B): 90% nu au, 10% au
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.9], [0.1]])

# Tusea (T) conditionata de Boala (B)
cpd_t = TabularCPD(variable='T', variable_card=2, 
                   values=[[0.7, 0.2], 
                           [0.3, 0.8]],
                   evidence=['B'], evidence_card=[2])

# Radiografia (X) conditionata de Boala (B)
cpd_x = TabularCPD(variable='X', variable_card=2, 
                   values=[[0.9, 0.1], 
                           [0.1, 0.9]],
                   evidence=['B'], evidence_card=[2])

# Dificultatea (D) conditionata de Boala (B) si Tuse (T)
cpd_d = TabularCPD(variable='D', variable_card=2, 
                   values=[[0.9, 0.5, 0.6, 0.1], 
                           [0.1, 0.5, 0.4, 0.9]],
                   evidence=['B', 'T'], evidence_card=[2, 2])

model.add_cpds(cpd_b, cpd_t, cpd_x, cpd_d)
model.check_model()

infer = VariableElimination(model)

# b) P(B=1 | T=1, D=1) 
rez_b = infer.query(['B'], evidence={'T': 1, 'D': 1})
print("b) Probabilitatea pacientului de a avea boala (T=1, D=1):")
print(rez_b)

# c) P(X=1 | B=0)
rez_c = infer.query(['X'], evidence={'B': 0})
print("\nc) Probabilitatea radiografiei anormale cand nu are boala (B=0):")
print(rez_c)