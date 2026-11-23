import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.stats import hypergeom

def calculeaza_hdi(esantioane, prob=0.95):
    """Calculeaza cel mai scurt interval (HDI) care contine `prob` din esantioane."""
    x = np.sort(np.asarray(esantioane).flatten())
    n_tot = len(x)
    k = int(np.floor(prob * n_tot))
    latimi = x[k:] - x[:n_tot - k]
    idx_min = np.argmin(latimi)
    return x[idx_min], x[idx_min + k]

# Datele observate
M = 20      # animale marcate initial
n = 15      # animale capturate a doua oara
Y_obs = 4   # animale marcate regasite

# Setari MCMC (mentionat in README.md)
DRAWS = 4000
TUNE = 2000
CHAINS = 4
SEED = 42

# a) Construirea modelului in PyMC si esantionarea posteriorului lui N
with pm.Model() as pop_model:
    N = pm.DiscreteUniform("N", lower=20, upper=200)
    # In PyMC: N = populatia totala, k = nr. elemente marcate (M), n = nr. extrageri
    Y = pm.HyperGeometric("Y", N=N, k=M, n=n, observed=Y_obs)

    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=1,
        random_seed=SEED
    )

    # d) Distributia predictiva a posteriori pentru o noua captura de 15 animale
    ppc = pm.sample_posterior_predictive(idata, random_seed=SEED)

# Extragem esantioanele intr-un singur vector 1D
N_samples = idata.posterior["N"].values.flatten()
Y_pred_samples = ppc.posterior_predictive["Y"].values.flatten()

# Diagnostice si rezumat 
print("=== Rezumat PyMC (HDI 95%) ===")
print(az.summary(idata, var_names=["N"], ci_prob=0.95))

# b) Media, mediana, HDI 95% si comparatia cu estimatorul clasic Mn/Y
media_N = np.mean(N_samples)
mediana_N = np.median(N_samples)
hdi_jos, hdi_sus = calculeaza_hdi(N_samples, prob=0.95)
estimare_clasica = (M * n) / Y_obs

print(f"\nb) Statistici a posteriori pentru N:")
print(f"   Media a posteriori:     {media_N:.2f}")
print(f"   Mediana a posteriori:   {mediana_N:.2f}")
print(f"   Interval HDI 95%:       [{hdi_jos}, {hdi_sus}]")
print(f"   Estimarea clasica Mn/Y: {estimare_clasica:.2f}")

# c) Probabilitatea a posteriori P(N > 100 | Y = 4)
prob_N_gt_100 = np.mean(N_samples > 100)
print(f"\nc) P(N > 100 | Y = 4) = {prob_N_gt_100:.4f} ({prob_N_gt_100*100:.2f}%)")

# Verificare exacta 
N_grid = np.arange(20, 201)
lik = hypergeom.pmf(Y_obs, N_grid, M, n)
post_exact = lik / np.sum(lik)
media_exacta = np.sum(N_grid * post_exact)
prob_exacta_gt_100 = np.sum(post_exact[N_grid > 100])
print(f"\n[Verificare exacta] Media N = {media_exacta:.2f}, P(N > 100 | Y = 4) = {prob_exacta_gt_100:.4f}")

# Statistici pentru distributia predictiva a posteriori (pct. d)
hdi_y_jos, hdi_y_sus = calculeaza_hdi(Y_pred_samples, prob=0.95)
print(f"\nd) Distributia predictiva a posteriori pentru Y_nou (n=15):")
print(f"   Media predictiva a animalelor marcate: {np.mean(Y_pred_samples):.2f}")
print(f"   HDI 95% pentru Y_nou: [{hdi_y_jos}, {hdi_y_sus}]")

# GRAFIC 1: Distributia a posteriori a lui N 
plt.figure(figsize=(10, 5))
plt.hist(N_samples, bins=np.arange(20, 202) - 0.5, density=True,
         alpha=0.65, color="steelblue", edgecolor="white", label="Posterior PyMC (MCMC)")
plt.plot(N_grid, post_exact, color="darkred", linewidth=2, label="Posterior exact")
plt.axvline(media_N, color="black", linestyle="--", linewidth=1.5, label=f"Media = {media_N:.1f}")
plt.axvline(mediana_N, color="green", linestyle=":", linewidth=1.8, label=f"Mediana = {mediana_N:.1f}")
plt.axvline(estimare_clasica, color="orange", linestyle="-.", linewidth=1.8, label=f"Clasic Mn/Y = {estimare_clasica:.1f}")
plt.title("Exercitiul 1: Distributia a posteriori a dimensiunii populatiei N")
plt.xlabel("Dimensiunea populatiei (N)")
plt.ylabel("Probabilitate a posteriori")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ex1_posterior_N.png", dpi=150)
plt.show()

# GRAFIC 2: Distributia predictiva a posteriori pentru Y_nou 
vals, counts = np.unique(Y_pred_samples, return_counts=True)
probs_pred = counts / len(Y_pred_samples)

plt.figure(figsize=(9, 4.5))
plt.bar(vals, probs_pred, color="teal", alpha=0.75, edgecolor="black")
plt.axvline(Y_obs, color="red", linestyle="--", linewidth=1.8, label=f"Observatia initiala (Y = {Y_obs})")
plt.title("Exercitiul 1: Distributia predictiva a posteriori pentru o noua captura (n = 15)")
plt.xlabel("Numarul de animale marcate intr-o captura nouă (Y_nou)")
plt.ylabel("Probabilitate predictiva")
plt.xticks(range(0, 16))
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ex1_predictive_Y.png", dpi=150)
plt.show()