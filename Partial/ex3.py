import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Prior initial: Beta(2, 8)
a_prior, b_prior = 2, 8

#  sub a) 5 defecte din 20 produse
n1, y1 = 20, 5
a1 = a_prior + y1
b1 = b_prior + (n1 - y1)
media1 = a1 / (a1 + b1)

print(f"(a) Posterior dupa lotul 1: Beta({a1}, {b1})")
print(f"    Media a posteriori = {media1:.4f}")

# Reprezentare grafica prior vs posterior a)
x = np.linspace(0, 1, 400)
prior_pdf = stats.beta(a_prior, b_prior).pdf(x)
post1_pdf = stats.beta(a1, b1).pdf(x)

plt.figure(figsize=(8, 5))
plt.plot(x, prior_pdf, label=f"Prior Beta({a_prior}, {b_prior})")
plt.plot(x, post1_pdf, label=f"Posterior Beta({a1}, {b1})")
plt.xlabel("theta")
plt.ylabel("Densitate")
plt.title("Densitatea a priori vs. a posteriori (lotul 1)")
plt.legend()
plt.savefig("subiect3_a.png")
plt.show()

# sub b) Al doilea 3 defecte din 15 produse 
n2, y2 = 15, 3
a2 = a1 + y2
b2 = b1 + (n2 - y2)
dist2 = stats.beta(a2, b2)

print(f"\n(b) Posterior dupa ambele loturi: Beta({a2}, {b2})")
print(f"    Media = {dist2.mean():.4f}, Dispersia = {dist2.var():.5f}")

# sub c) Doar al doilea lot cu priorul initial Beta(2, 8)
a3 = a_prior + y2
b3 = b_prior + (n2 - y2)
dist3 = stats.beta(a3, b3)

print(f"\n(c) Posterior doar lotul 2: Beta({a3}, {b3})")
print(f"    Media = {dist3.mean():.4f}, Dispersia = {dist3.var():.5f}")