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

# Datele observate (30 de zile)
Y_obs = np.array([
    3, 4, 2, 5, 3, 4, 3, 2, 4, 3,
    4, 5, 5, 6, 5, 7, 6, 8, 7, 9,
    8, 10, 9, 8, 11, 9, 10, 8, 9, 10
])
t_zile = np.arange(1, 31)

# Setari MCMC (README.md)
DRAWS = 2000
TUNE = 2000
CHAINS = 4
SEED = 42

# a) Definirea modelului in PyMC si esantionarea posteriorului
with pm.Model() as tranzitie_model:
    lam1 = pm.Exponential("lam1", lam=0.2)
    lam2 = pm.Exponential("lam2", lam=0.2)
    tau = pm.Uniform("tau", lower=1.0, upper=30.0)
    k = pm.Exponential("k", lam=1.0)

    # Rata zilnica: lam1 + (lam2 - lam1) / (1 + exp(-k * (t - tau)))
    lam_t = pm.Deterministic(
        "lam_t",
        lam1 + (lam2 - lam1) * pm.math.sigmoid(k * (t_zile - tau))
    )

    obs = pm.Poisson("obs", mu=lam_t, observed=Y_obs)

    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=1,
        target_accept=0.9,
        random_seed=SEED
    )

    # d) Distributia predictiva a posteriori pentru cele 30 de zile
    ppc = pm.sample_posterior_predictive(idata, random_seed=SEED)

# Rezumat si diagnostice (r_hat, ESS, HDI 95%)
print("=== Rezumat PyMC (HDI 95%) ===")
print(az.summary(idata, var_names=["lam1", "lam2", "tau", "k"], ci_prob=0.95))

# Extragem esantioanele posterioare
lam1_samples = idata.posterior["lam1"].values.flatten()
lam2_samples = idata.posterior["lam2"].values.flatten()
tau_samples = idata.posterior["tau"].values.flatten()
k_samples = idata.posterior["k"].values.flatten()

# b) Statistici si HDI 95% pentru tau si k
hdi_tau = calculeaza_hdi(tau_samples, prob=0.95)
hdi_k = calculeaza_hdi(k_samples, prob=0.95)
hdi_lam1 = calculeaza_hdi(lam1_samples, prob=0.95)
hdi_lam2 = calculeaza_hdi(lam2_samples, prob=0.95)

print("\nb) Estimari a posteriori (media si HDI 95%):")
print(f"   lam1 (nivel initial): media = {np.mean(lam1_samples):.2f}, HDI 95% = [{hdi_lam1[0]:.2f}, {hdi_lam1[1]:.2f}]")
print(f"   lam2 (nivel final):   media = {np.mean(lam2_samples):.2f}, HDI 95% = [{hdi_lam2[0]:.2f}, {hdi_lam2[1]:.2f}]")
print(f"   tau  (centru tranz.): media = {np.mean(tau_samples):.2f}, HDI 95% = [{hdi_tau[0]:.2f}, {hdi_tau[1]:.2f}]")
print(f"   k    (viteza):        media = {np.mean(k_samples):.2f}, HDI 95% = [{hdi_k[0]:.2f}, {hdi_k[1]:.2f}]")

# c) Probabilitatea a posteriori P(lam2 > lam1 | Y)
prob_crestere = np.mean(lam2_samples > lam1_samples)
print(f"\nc) P(lam2 > lam1 | Y_1,...,Y_30) = {prob_crestere:.4f} ({prob_crestere*100:.2f}%)")

# GRAFIC 1 (pct b): Distributiile a posteriori pentru tau si k 
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(tau_samples, bins=50, density=True, color="steelblue", alpha=0.7, edgecolor="white")
axes[0].axvline(np.mean(tau_samples), color="black", linestyle="--", label=f"Media = {np.mean(tau_samples):.2f}")
axes[0].hlines(y=0.005, xmin=hdi_tau[0], xmax=hdi_tau[1], color="red", linewidth=4,
               label=f"HDI 95%: [{hdi_tau[0]:.2f}, {hdi_tau[1]:.2f}]")
axes[0].set_title("Posterior pentru tau (momentul central al tranzitiei)")
axes[0].set_xlabel("Ziua (tau)")
axes[0].set_ylabel("Densitate")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(k_samples, bins=50, density=True, color="darkorange", alpha=0.7, edgecolor="white")
axes[1].axvline(np.mean(k_samples), color="black", linestyle="--", label=f"Media = {np.mean(k_samples):.2f}")
axes[1].hlines(y=0.02, xmin=hdi_k[0], xmax=hdi_k[1], color="red", linewidth=4,
               label=f"HDI 95%: [{hdi_k[0]:.2f}, {hdi_k[1]:.2f}]")
axes[1].set_title("Posterior pentru k (rapiditatea schimbarii)")
axes[1].set_xlabel("Panta tranzitiei (k)")
axes[1].set_ylabel("Densitate")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ex3_posterior_tau_k.png", dpi=150)
plt.show()

# GRAFIC 2 (pct c): Evolutia estimata a lui lambda_t si datele observate 
lam_t_samples = idata.posterior["lam_t"].values.reshape(-1, len(t_zile))
lam_t_mean = np.mean(lam_t_samples, axis=0)
lam_t_hdi = np.array([calculeaza_hdi(lam_t_samples[:, i], prob=0.95) for i in range(len(t_zile))])

plt.figure(figsize=(10, 5))
plt.scatter(t_zile, Y_obs, color="black", zorder=5, label="Date observate (Y_t)")
plt.plot(t_zile, lam_t_mean, color="darkred", linewidth=2.2, label="Media a posteriori a lui lambda_t")
plt.fill_between(t_zile, lam_t_hdi[:, 0], lam_t_hdi[:, 1], color="darkred", alpha=0.22,
                 label="HDI 95% pentru lambda_t")
plt.axvline(np.mean(tau_samples), color="steelblue", linestyle="--", linewidth=1.5,
            label=f"Media tau = {np.mean(tau_samples):.1f}")
plt.title("Exercitiul 3: Evolutia estimata a ratei lambda_t fata de datele observate")
plt.xlabel("Ziua (t)")
plt.ylabel("Numar solicitari zilnice")
plt.xticks(range(1, 31, 2))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ex3_evolutie_lambda.png", dpi=150)
plt.show()

# GRAFIC 3 (pct d): Distributia predictiva a posteriori pentru cele 30 de zile 
y_pred_samples = ppc.posterior_predictive["obs"].values.reshape(-1, len(t_zile))
y_pred_mean = np.mean(y_pred_samples, axis=0)
y_pred_low = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_high = np.percentile(y_pred_samples, 97.5, axis=0)

in_interval = np.sum((Y_obs >= y_pred_low) & (Y_obs <= y_pred_high))
print(f"\nd) Verificare predictiva a posteriori:")
print(f"   Zile observate aflate in intervalul predictiv 95%: {in_interval} din {len(t_zile)} ({in_interval/len(t_zile)*100:.1f}%)")

plt.figure(figsize=(10, 5))
plt.fill_between(t_zile, y_pred_low, y_pred_high, color="teal", alpha=0.22,
                 label="Interval predictiv 95% (Y_nou)")
plt.plot(t_zile, y_pred_mean, color="teal", linewidth=2, linestyle="--",
         label="Media predictiva a posteriori")
plt.scatter(t_zile, Y_obs, color="black", zorder=5, label="Date observate (Y_t)")
plt.title("Exercitiul 3: Distributia predictiva a posteriori vs. datele observate")
plt.xlabel("Ziua (t)")
plt.ylabel("Numar solicitari zilnice")
plt.xticks(range(1, 31, 2))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ex3_posterior_predictive.png", dpi=150)
plt.show()