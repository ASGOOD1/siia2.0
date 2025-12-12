import pandas as pd
import numpy as np

np.random.seed(42)

n = 10000

# environment
env = np.random.choice(["urban", "rural"], size=n)


media = np.random.normal(loc=7, scale=2, size=n).astype(float)
mask = np.random.rand(n) < 0.05
media[mask] = np.nan
media = np.clip(media, 0, 10)


absente = np.random.poisson(lam=5, size=n).astype(float)
mask = np.random.rand(n) < 0.05
absente[mask] = np.nan
absente = np.clip(absente, 0, 100)


comportament = np.random.rand(n)
mask = np.random.rand(n) < 0.05
comportament[mask] = np.nan

cond_strong = (np.nan_to_num(media, nan=5) > 5) & (np.nan_to_num(absente, nan=45) < 30)


cond_weak = (np.nan_to_num(media, nan=3) > 3) & (np.nan_to_num(absente, nan=50) < 50)

trece = np.zeros_like(media, dtype=int)

trece[cond_strong] = 1

trece[cond_weak & ~cond_strong] = np.random.randint(0, 2, size=(cond_weak & ~cond_strong).sum())

df = pd.DataFrame({
    "environment": env,
    "media_s1": np.round(media, 2),
    "absente": absente,
    "comportament": np.round(comportament, 2),
    "trece": trece
})


df.to_csv("elevi_10000.csv", index=False)
print("CSV cu 10000 de rânduri creat: elevi_10000.csv")