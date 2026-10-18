from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definim structura grafului
# C: Usa aleasa de concurent
# P: Usa unde se afla premiul
# H: Usa deschisa de prezentator (Monty)
model = BayesianNetwork([('C', 'H'), ('P', 'H')])

# 2. Definim probabilitatile de baza
# Concurentul alege o usa absolut la intamplare (sanse 1/3 pentru fiecare)
cpd_c = TabularCPD(variable='C', variable_card=3, values=[[1/3], [1/3], [1/3]])

# Premiul e ascuns la intamplare in spatele unei usi (sanse 1/3 pentru fiecare)
cpd_p = TabularCPD(variable='P', variable_card=3, values=[[1/3], [1/3], [1/3]])

# 3. Definim regulile lui Monty (Tabelul Conditionat pentru nodul H)
# Regulile jocului:
# - Monty nu deschide usa aleasa de concurent (C).
# - Monty nu deschide usa cu premiul (P).
# - Daca C si P sunt la fel (ai nimerit premiul din prima), Monty alege la intamplare din cele 2 usi ramase (0.5 si 0.5 sanse).
cpd_h = TabularCPD(variable='H', variable_card=3, 
                   values=[[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                           [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                           [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]],
                   evidence=['C', 'P'], evidence_card=[3, 3])

# Adaugam tabelele in retea si verificam daca structura e corecta
model.add_cpds(cpd_c, cpd_p, cpd_h)
model.check_model() 

# 4. Formulam interogarea (Simularea jocului)
infer = VariableElimination(model)

# Scenariul nostru: Noi alegem usa 0. Monty deschide usa 2. Unde e premiul?
posterior_p = infer.query(['P'], evidence={'C': 0, 'H': 2})

print("Probabilitatile pentru locatia premiului, stiind ca am ales usa 0 si s-a deschis usa 2:\n")
print(posterior_p)

# EXPLICATIA PROBLEMEI:
# Problema Monty Hall este un paradox celebru bazat pe un joc televizat.
# Imagineaza-ti ca esti la un concurs. In fata ta sunt 3 usi: in spatele 
# uneia este o masina (premiul), iar in spatele celorlalte doua sunt capre.
# 
# Tu alegi o usa (sa zicem Usa 0). Prezentatorul (Monty), care stie exact 
# unde e masina, deschide o alta usa (sa zicem Usa 2) ca sa iti arate o capra. 
# Apoi te intreaba: "Vrei sa ramai la usa ta sau vrei sa schimbi cu usa ramasa inchisa?"
# 
# Intuitia ne spune ca daca au mai ramas 2 usi, sansele sunt 50-50, deci nu conteaza. 
# In realitate, matematica demonstreaza ca daca iti schimbi alegerea initiala, 
# iti dublezi sansele de castig (de la 33% la 66%).
#
# EXPLICATIA CODULUI (IMPLEMENTAREA):
# Codul modeleaza acest joc creand o Retea Bayesiana cu 3 variabile (noduri):
# 1. C (Alegerea concurentului)
# 2. P (Locatia premiului)
# 3. H (Alegerea prezentatorului)
# 
# Alegerea ta (C) si locatia masinii (P) sunt intamplatoare (sanse de 1/3 fiecare) 
# si nu depind una de alta. Insa usa pe care o deschide Monty (H) depinde direct 
# si de ce ai ales tu, si de unde e masina. De aceea sagetile merg de la C si P spre H.
#
# Tabelul lui Monty (cpd_h din cod) forteaza programul sa respecte regulile jocului:
# - Monty are probabilitate 0 sa deschida usa aleasa de tine.
# - Monty are probabilitate 0 sa deschida usa cu masina.
# - Daca ai nimerit masina din prima, el alege la intamplare (0.5 si 0.5) din cele 2 usi ramase.
# 
# La final, folosim "VariableElimination" pentru a afla raspunsul. Ii spunem calculatorului: 
# "Stim ca am ales usa 0 si Monty a deschis usa 2". Reteaua va calcula automat 
# si ne va demonstra ca usa 1 are acum 66% sanse sa ascunda masina.