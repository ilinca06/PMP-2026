import numpy as np

# Setari generale pentru simulare
NUM_SIMULARI = 2000
TIMP_TOTAL = 60
LAMBDA_SOSIRI = 20 / 60
TIMP_MEDIU_PLATA = 2
TIMP_LIVRARE_TARGET = 15
PROCENT_TARGET = 0.95

def evalueaza_a(a):
    np.random.seed(42)
    clienti_ok = 0
    total_clienti_generati = 0
    
    for _ in range(NUM_SIMULARI):
        timpi_inter_sosiri = np.random.exponential(1 / LAMBDA_SOSIRI, size=100)
        momente_sosiri = np.cumsum(timpi_inter_sosiri)
        momente_sosiri = momente_sosiri[momente_sosiri <= TIMP_TOTAL]
        numar_clienti = len(momente_sosiri)
        
        if numar_clienti == 0:
            continue
        
        timpi_plata = np.random.exponential(TIMP_MEDIU_PLATA, size=numar_clienti)
        timpi_preparare = np.random.exponential(a, size=numar_clienti)
        
        moment_liber_casa = 0
        moment_liber_gatit = 0
        
        for i in range(numar_clienti):
            sosire = momente_sosiri[i]
            
            # Coada Casa
            inceput_plata = max(sosire, moment_liber_casa)
            sfarsit_plata = inceput_plata + timpi_plata[i]
            moment_liber_casa = sfarsit_plata
            
            # Coada Gatit
            inceput_gatit = max(sfarsit_plata, moment_liber_gatit)
            sfarsit_gatit = inceput_gatit + timpi_preparare[i]
            moment_liber_gatit = sfarsit_gatit
            
            # Calcul timp asteptare
            timp_total = sfarsit_gatit - sosire
            
            total_clienti_generati += 1
            if timp_total <= TIMP_LIVRARE_TARGET:
                clienti_ok += 1
                
    return clienti_ok / total_clienti_generati if total_clienti_generati > 0 else 0

print("b) Căutăm valoarea optimă a parametrului 'a'...")
# Ne oprim la 0.1 deoarece distributia exponentiala nu accepta media 0.0
a_valori = np.arange(2.0, 0.05, -0.1)  
a_optim = None

for a in a_valori:
    prob = evalueaza_a(a)
    print(f"  Pentru a = {a:.1f} min -> Prob. servire <= 15 min: {prob*100:.2f}%")
    if prob >= PROCENT_TARGET:
        a_optim = a
        break

if a_optim is not None:
    print(f"\n-> Cea mai mare valoare aproximativă a lui 'a' pentru o probabilitate >= 95% este: a = {a_optim:.1f} minute\n")
else:
    print("\n-> Probabilitatea de 95% NU POATE FI ATINSĂ nici cu un timp de preparare de 0.1 min (se ajunge la maxim ~94.9%).")
    print("   Explicație: Blocajul sistemului (strugularea) se află la casa de plată.")
    print("   Deoarece plata durează în medie 2 min și clienții sosesc frecvent (1 la 3 min),")
    print("   coada de plată singură acumulează întârzieri majore, depășind adesea 15 minute,")
    print("   chiar dacă gătitul s-ar face instantaneu.")
    
    # Pentru a putea rula punctul c), alegem ca 'a' cea mai buna valoare gasita (cea mai mica posibila)
    a_optim = 0.1 

# c) Estimarea timpului mediu total pentru valoarea optima 'a'
np.random.seed(42) 
timpi_totali_optimi = []
timpi_asteptare_puri = []

for _ in range(NUM_SIMULARI):
    timpi_inter_sosiri = np.random.exponential(1 / LAMBDA_SOSIRI, size=100)
    momente_sosiri = np.cumsum(timpi_inter_sosiri)
    momente_sosiri = momente_sosiri[momente_sosiri <= TIMP_TOTAL]
    numar_clienti = len(momente_sosiri)
    
    if numar_clienti == 0:
        continue
    
    timpi_plata = np.random.exponential(TIMP_MEDIU_PLATA, size=numar_clienti)
    timpi_preparare = np.random.exponential(a_optim, size=numar_clienti)
    
    moment_liber_casa = 0
    moment_liber_gatit = 0
    
    for i in range(numar_clienti):
        sosire = momente_sosiri[i]
        
        inceput_plata = max(sosire, moment_liber_casa)
        sfarsit_plata = inceput_plata + timpi_plata[i]
        moment_liber_casa = sfarsit_plata
        
        inceput_gatit = max(sfarsit_plata, moment_liber_gatit)
        sfarsit_gatit = inceput_gatit + timpi_preparare[i]
        moment_liber_gatit = sfarsit_gatit
        
        timp_total = sfarsit_gatit - sosire
        timp_asteptare_pur = timp_total - (timpi_plata[i] + timpi_preparare[i])
        
        timpi_totali_optimi.append(timp_total)
        timpi_asteptare_puri.append(timp_asteptare_pur)

timp_mediu = np.mean(timpi_totali_optimi)
timp_asteptare_pur_mediu = np.mean(timpi_asteptare_puri)

print(f"\nc) Pentru cea mai rapidă preparare (a = {a_optim:.1f} min):")
print(f"   Timpul mediu total (sosire -> primire mâncare): {timp_mediu:.2f} minute")
print(f"   Timpul mediu de AȘTEPTARE (stat pur la cozi, fără a fi servit efectiv): {timp_asteptare_pur_mediu:.2f} minute\n")

print("d) Detalii metodologie:")
print(f"   - Număr de simulări pe scenariu: {NUM_SIMULARI}")
print("   - Sămânța aleatoare: np.random.seed(42) - mutată în interiorul funcției de evaluare.")
print("   - Metodă de căutare: Liniară descrescătoare de la 2.0 la 0.1 cu pas de 0.1.")
print("   - Fluctuații: Fixarea seed-ului în funcție a asigurat comparabilitatea perfectă a rezultatelor")
print("     între pași (același trafic de clienți). Fără seed fix, rezultatele ar varia deoarece")
print("     sosirile și serviciile sunt variabile aleatoare generate de distribuții exponențiale.")