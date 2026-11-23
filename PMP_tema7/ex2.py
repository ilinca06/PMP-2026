import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

def calculeaza_hdi(esantioane, prob=0.95):
    """Calculeaza cel mai scurt interval (HDI) care contine `prob` din esantioane."""
    x = np.sort(np.asarray(esantioane).flatten())
    n_tot = len(x)
    k = int(np.floor(prob * n_tot))
    latimi = x[k:] - x[:n_tot - k]
    idx_min = np.argmin(latimi)
    return x[idx_min], x[idx_min + k]

# Datele observate (timpi de recuperare in zile)
A_obs = np.array([3.2, 4.1, 2.8, 5.0, 3.7, 4.5, 3.0, 3.9])
B_obs = np.array([2.1, 2.8, 3.4, 2.5, 3.0, 2.2, 3.6, 2.7])

# Setari MCMC (README.md)
DRAWS = 4000
TUNE = 2000
CHAINS = 4
SEED = 42

# a) Definirea modelului in PyMC si esantionarea posteriorului
with pm.Model() as tratamente_model:
    # Priori Gamma(alpha=2, beta=4) in parametrizarea forma-rata
    lam_A = pm.Gamma("lam_A", alpha=2, beta=4)
    lam_B = pm.Gamma("lam_B", alpha=2, beta=4)

    # Timpii medii de recuperare: mu = 1 / lambda
    mu_A = pm.Deterministic("mu_A", 1.0 / lam_A)
    mu_B = pm.Deterministic("mu_B", 1.0 / lam_B)

    # Verosimilitati Exponential(lambda)
    obs_A = pm.Exponential("obs_A", lam=lam_A, observed=A_obs)
    obs_B = pm.Exponential("obs_B", lam=lam_B, observed=B_obs)

    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=1,
        random_seed=SEED
    )

    # d) Distributii predictive a posteriori pentru pacienti noi
    ppc = pm.sample_posterior_predictive(idata, random_seed=SEED)

# Rezumat si diagnostice (r_hat, ESS, HDI 95%)
print("Rezumat PyMC (HDI 95%)")
print(az.summary(idata, var_names=["lam_A", "lam_B", "mu_A", "mu_B"], ci_prob=0.95))

# Extragem esantioanele posterioare ale timpilor medii
mu_A_samples = idata.posterior["mu_A"].values.flatten()
mu_B_samples = idata.posterior["mu_B"].values.flatten()

# b) Statistici si intervale HDI de 95% pentru 1/lam_A si 1/lam_B
hdi_A = calculeaza_hdi(mu_A_samples, prob=0.95)
hdi_B = calculeaza_hdi(mu_B_samples, prob=0.95)

print("\nb) Timpii medii de recuperare (a posteriori):")
print(f"   Tratament A (1/lam_A): media = {np.mean(mu_A_samples):.3f} zile, HDI 95% = [{hdi_A[0]:.3f}, {hdi_A[1]:.3f}]")
print(f"   Tratament B (1/lam_B): media = {np.mean(mu_B_samples):.3f} zile, HDI 95% = [{hdi_B[0]:.3f}, {hdi_B[1]:.3f}]")

# c) Probabilitatea a posteriori ca timpul mediu pentru B sa fie mai mic decat pentru A
prob_medie_B_mai_mic = np.mean(mu_B_samples < mu_A_samples)
print(f"\nc) P(1/lam_B < 1/lam_A | date) = {prob_medie_B_mai_mic:.4f} ({prob_medie_B_mai_mic*100:.2f}%)")

# d) Probabilitatea predictiva ca un pacient nou cu B sa se recupereze mai repede decat unul cu A
pred_A = ppc.posterior_predictive["obs_A"].values.flatten()
pred_B = ppc.posterior_predictive["obs_B"].values.flatten()
prob_pacient_B_mai_repede = np.mean(pred_B < pred_A)

print(f"\nd) P(X_nou_B < X_nou_A | date) = {prob_pacient_B_mai_repede:.4f} ({prob_pacient_B_mai_repede*100:.2f}%)")

# GRAFIC 1: Distributiile a posteriori ale timpilor medii 1/lam_A si 1/lam_B 
plt.figure(figsize=(10, 5))
bins = np.linspace(0, 10, 80)
plt.hist(mu_A_samples, bins=bins, density=True, alpha=0.55, color="steelblue", label=f"1/λ_A (media={np.mean(mu_A_samples):.2f})")
plt.hist(mu_B_samples, bins=bins, density=True, alpha=0.55, color="darkorange", label=f"1/λ_B (media={np.mean(mu_B_samples):.2f})")

# Marcam intervalele HDI 95% pe axa de jos
plt.hlines(y=0.015, xmin=hdi_A[0], xmax=hdi_A[1], color="steelblue", linewidth=4, label=f"HDI 95% A: [{hdi_A[0]:.2f}, {hdi_A[1]:.2f}]")
plt.hlines(y=0.045, xmin=hdi_B[0], xmax=hdi_B[1], color="darkorange", linewidth=4, label=f"HDI 95% B: [{hdi_B[0]:.2f}, {hdi_B[1]:.2f}]")

plt.title("Exercitiul 2: Distributiile a posteriori ale timpilor medii de recuperare")
plt.xlabel("Timp mediu de recuperare (zile)")
plt.ylabel("Densitate a posteriori")
plt.xlim(0, 10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ex2_posterior_medii.png", dpi=150)
plt.show()

# GRAFIC 2: Distributiile predictive a posteriori pentru un pacient nou
plt.figure(figsize=(10, 5))
bins_pred = np.linspace(0, 15, 75)
plt.hist(pred_A, bins=bins_pred, density=True, alpha=0.5, color="steelblue", label="Pacient nou - Tratament A")
plt.hist(pred_B, bins=bins_pred, density=True, alpha=0.5, color="darkorange", label="Pacient nou - Tratament B")
plt.title(f"Exercițiul 2: Distributiile predictive pentru un pacient nou (P(B < A) = {prob_pacient_B_mai_repede*100:.1f}%)")
plt.xlabel("Timp de recuperare prezis (zile)")
plt.ylabel("Densitate predictiva")
plt.xlim(0, 15)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ex2_predictive_pacienti.png", dpi=150)
plt.show()