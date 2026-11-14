import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent

L, a, delta_G = 1.0, 1.0, 0.0
controls = [0.0, 1/6, 1/3, 0.5]  # assumed redox-induced energy biases
kBT, D = 0.025, 0.005
T, dt, N = 20.0, 0.01, 4000
target = 0.9 * L                  #survival prob
rng = np.random.default_rng(2026)
time = np.arange(int(T / dt) + 1) * dt


def G(c, u):
    return a * c**2 * (c - L)**2 + (delta_G - u) * c / L - u


def dG(c, u):
    return 2 * a * c * (c - L) * (2 * c - L) + (delta_G - u) / L


fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
results = []
output = [time]
space = np.linspace(-0.4, 1.2, 500)
bins_c = np.linspace(-0.45, target, 70)
bins_t = np.linspace(0, T, 41)

for u in controls:
    c = np.zeros(N)                    # TTFstate
    switched = np.zeros(N, dtype=bool) # absorbing boundary
    arrival = np.full(N, np.inf)       # first-passage times
    P = np.zeros(len(time))            # P(t)

    for step in range(1, len(time)):
        alive = ~switched
        c[alive] += -(D / kBT) * dG(c[alive], u) * dt \
                    + np.sqrt(2 * D * dt) * rng.normal(size=alive.sum())
        newly = alive & (c >= target)
        arrival[newly] = time[step]
        switched[newly] = True
        P[step] = switched.mean()

    S = 1 - P                         
    f = np.histogram(arrival[np.isfinite(arrival)], bins=bins_t)[0] / (N * np.diff(bins_t))
    p = np.histogram(c[~switched], bins=bins_c)[0] / (N * np.diff(bins_c))
    confidence = 1.96 * np.sqrt(P[-1] * (1 - P[-1]) / N)
    restricted_mean = dt * (S.sum() - (S[0] + S[-1]) / 2)  # E[min(tau, T)]
    results.append([u, P[-1], confidence, restricted_mean])
    output.extend([P, S])

    name = f'u = {u:.3f}'
    ax[0, 0].plot(space, G(space, u) - G(0, u), label=name)
    ax[0, 1].plot((bins_c[:-1] + bins_c[1:]) / 2, p, label=name)
    ax[1, 0].plot(time, P, label=name)
    ax[1, 1].plot((bins_t[:-1] + bins_t[1:]) / 2, f, label=name)

ax[0, 0].set(title='Controlled potential: G(c,u)', xlabel='Reaction coordinate c', ylabel='G(c,u) − G(0,u)')
ax[0, 1].set(title='Surviving probability density at T', xlabel='Reaction coordinate c', ylabel='p(c,T), before first passage')
ax[1, 0].set(title='First-passage switching probability: 1 − S(t)', xlabel='Reduced time', ylabel='P(switch by t)', ylim=(0, 1.02))
ax[1, 1].set(title='First-passage time density: −dS/dt', xlabel='Reduced time', ylabel='f(t)')
for panel in ax.flat:
    panel.grid(alpha=0.2)
    panel.legend(fontsize=8)

fig.savefig(BASE / 'controlledpotential_first_passage.png', dpi=200)
np.savetxt(BASE / 'first_passage_curves.csv', np.column_stack(output), delimiter=',',
           header='time,' + ','.join(f'P_u{u:.3f},S_u{u:.3f}' for u in controls), comments='')
np.savetxt(BASE / 'first_passage_summary.csv', results, delimiter=',',
           header='u,P_switch_at_T,CI95_halfwidth,restricted_mean_time', comments='')

for u, P_end, ci, mean in results:
    print(f'u={u:.3f}: P_switch({T:g})={P_end:.1%} ± {ci:.1%} (approx. 95% CI), '
          f'E[min(tau,{T:g})]={mean:.2f}')
plt.close(fig)
print(f'\nSaved plots and CSV files to: {BASE}')
if os.name == 'nt':
    os.startfile(str(BASE / 'controlledpotential_first_passage.png'))
