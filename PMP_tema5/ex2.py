import numpy as np

# a) Definirea modelului
# Ordinea starilor: 0=Dificil, 1=Mediu, 2=Usor
stari = ["Dificil", "Mediu", "Usor"]
# Ordinea observatiilor: 0=FB, 1=B, 2=S, 3=NS
observatii = ["FB", "B", "S", "NS"]

# Distributia initiala (pi)
pi = np.array([1/3, 1/3, 1/3])

# Matricea de tranzitie A
A = np.array([
    [0.0, 0.5,  0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

# Matricea de emisie B
B = np.array([
    [0.1,  0.2,  0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2,  0.3,  0.4, 0.1]
])

# Verificam ca suma pe randuri sa fie 1
print("Suma randurilor din A:", np.sum(A, axis=1))
print("Suma randurilor din B:", np.sum(B, axis=1))

# Secventa observata (convertita in indecsi): FB, FB, S, B, B, S, B, B, NS, B, B, S
O = [0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1, 2]
T = len(O) # Numarul de observatii (12)
N = len(stari) # Numarul de stari (3)

# Tabele pentru Viterbi
# delta va tine minte probabilitatile maxime
delta = np.zeros((T, N))
# psi va tine minte indicele starii precedente (backtracking)
psi = np.zeros((T, N), dtype=int)

# b) Initializarea Viterbi
# Pentru primul pas (t=0, dar notat t=1 teoretic)
for j in range(N):
    delta[0, j] = pi[j] * B[j, O[0]]
    psi[0, j] = 0 # Nu exista stare precedenta

# c) Implementarea recurentei
for t in range(1, T):
    for j in range(N):
        # Calculam formula din paranteza pentru toate starile i anterioare
        scoruri = [delta[t-1, i] * A[i, j] for i in range(N)]
        
        # Extragem maximul din paranteza si il inmultim cu emisia
        delta[t, j] = B[j, O[t]] * max(scoruri)
        
        # argmax ne spune din ce stare anterioara provine maximul
        psi[t, j] = np.argmax(scoruri)

# Explicatie: Rolul lui psi_t(j) este de a păstra un "indicator" (un pointer) catre 
# starea de la momentul t-1 care a contribuit la probabilitatea maxima pentru 
# starea j de la momentul t. Acesta ne permite sa reconstituim traseul.

# d) Verificare intermediara
print("\nd) delta_1 (prob. dupa prima nota FB):")
print(delta[0])
print("\ndelta_2 (prob. dupa a doua nota FB):")
print(delta[1])
# Verificare manuala pentru delta_2(Mediu) = B(Mediu, FB) * max(delta_1(i) * A(i, Mediu))
# = 0.15 * max(0.033*0.5, 0.05*0.25, 0.066*0.25) = 0.15 * max(0.0166, 0.0125, 0.0166) = 0.0025

# e) Reconstructia traseului (Backtracking)
traseu = np.zeros(T, dtype=int)

# 1. Alegem starea finala cu cel mai mare scor
traseu[T-1] = np.argmax(delta[T-1])
scor_maxim = np.max(delta[T-1])

# 2. Mergem inapoi pas cu pas folosind indicatoarele din psi
for t in range(T-2, -1, -1):
    traseu[t] = psi[t+1, traseu[t+1]]

# Traducem indecsii in cuvinte
dificultati = [stari[idx] for idx in traseu]

print("\ne) Secventa cea mai probabila de dificultati este:")
print(dificultati)
print(f"Probabilitatea comuna maxima (scorul) este: {scor_maxim}")

# f) Interpretare
print("\nf) Interpretare:")
print("Viterbi caută cea mai probabilă secvență GLOBALĂ, deoarece alegerile")
print("optime făcute individual la fiecare pas ar putea produce un traseu invalid")
print("sau imposibil dpdv al matricei de tranziție (ex: două teste dificile la rând).")
print("Scorul afișat mai sus este P(O, X), adică probabilitatea comună de a observa")
print("acele note DACA s-ar fi dat fix acele teste (nu este P(X|O)).")