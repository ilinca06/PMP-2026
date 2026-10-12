# datele din problema
p_spam = 0.3
p_legitim = 0.7

# datele pentru cuvantul "oferta"
p_oferta_dat_spam = 0.8
p_oferta_dat_legitim = 0.15

# datele pentru cuvantul "gratuit"
p_gratuit_dat_spam = 0.6
p_gratuit_dat_legitim = 0.1


# 1. probabilitatea sa fie spam daca are cuvantul "oferta" 
# Formula Bayes: (P(Oferta|Spam) * P(Spam)) / Total probabilitate Oferta
numarator_1 = p_oferta_dat_spam * p_spam
numitor_1 = (p_oferta_dat_spam * p_spam) + (p_oferta_dat_legitim * p_legitim)

p_spam_dat_oferta = numarator_1 / numitor_1

print("\n1. Daca mesajul contine 'oferta':")
print("Probabilitatea de SPAM este:", round(p_spam_dat_oferta * 100, 2), "%")


# 2. probabilitatea sa fie spam daca are ambele cuvinte
# formula independentei conditionate
p_ambele_dat_spam = p_oferta_dat_spam * p_gratuit_dat_spam
p_ambele_dat_legitim = p_oferta_dat_legitim * p_gratuit_dat_legitim

# aplicam formula Bayes 
numarator_2 = p_ambele_dat_spam * p_spam
numitor_2 = (p_ambele_dat_spam * p_spam) + (p_ambele_dat_legitim * p_legitim)

p_spam_dat_ambele = numarator_2 / numitor_2

print("\n2. Daca mesajul contine si 'oferta' si 'gratuit':")
print("Probabilitatea de SPAM creste la:", round(p_spam_dat_ambele * 100, 2), "%")


# 4. exemplu de Regula de Clasificare 
print("\n 4. Testare regula decizie")
prag_decizie = 0.90 # setam pragul de 90% 

if p_spam_dat_ambele > prag_decizie:
    print("Sistemul a decis: Mesaj blocat! (Mutat in folderul SPAM)")
else:
    print("Sistemul a decis: Mesaj permis. (Ramane in Inbox)")