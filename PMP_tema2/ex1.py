import random

# facem 100.000 de aruncari 
N = 100000

# variabile in care vom numara de cate ori se intampla ceva
# pentru punctul 1
total_x_par = 0
fav_1 = 0

# pentru punctul 2
total_macar_un_par = 0
fav_2 = 0

# pentru punctul 3
total_suma_8 = 0
fav_3 = 0

# pentru punctul 4 
total_suma_10 = 0
total_x_par_si_suma_10 = 0

# aruncam zarurile de N ori
for i in range(N):
    x = random.randint(1, 6) # primul zar
    y = random.randint(1, 6) # al doilea zar
    suma = x + y
    
    # punctul 1 
    # daca primul zar e par
    if x % 2 == 0:
        total_x_par += 1
        # verificam daca suma >= 10
        if suma >= 10:
            fav_1 += 1
            total_x_par_si_suma_10 += 1 # il numaram si pentru punctul 4
            
    # punctul 2 
    if x % 2 == 0 or y % 2 == 0:
        total_macar_un_par += 1
        if suma >= 10:
            fav_2 += 1
            
    # punctul 3 
    if suma == 8:
        total_suma_8 += 1
        if x % 2 == 0 and y % 2 == 0:
            fav_3 += 1
            
    # punctul 4 (calculam cate au suma >= 10 in general) 
    if suma >= 10:
        total_suma_10 += 1


print("1. P(suma >= 10 | x e par) =", round(fav_1 / total_x_par, 4))
print("2. P(suma >= 10 | cel putin unul par) =", round(fav_2 / total_macar_un_par, 4))
print("3. P(ambele pare | suma = 8) =", round(fav_3 / total_suma_8, 4))

print("\n 4. Verificare independenta")
p_A = total_suma_10 / N
p_B = total_x_par / N
p_A_si_B = total_x_par_si_suma_10 / N

print("P(A) * P(B) =", round(p_A * p_B, 4))
print("P(A si B) =", round(p_A_si_B, 4))
print("Cele doua valori nu sunt egale, deci NU sunt independente.")