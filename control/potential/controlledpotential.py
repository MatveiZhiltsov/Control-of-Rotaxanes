import numpy as np
import matplotlib.pyplot as plt


L = 1.0
a = 1.0
delta_G = 0.0


c=np.linspace(-0.5, 1.5, 1000)
u_v=np.linspace(0, 0.5, 4)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Controlled Energy landscape for the instanteneously applied control input")
axes = axes.flatten()

for i, u in enumerate(u_v):
    G = a * c**2 * (c-L)**2 + (delta_G - u) * c / L - u
    ax = axes[i]
    ax.plot(c, G, color = 'blue')
    ax.set_title('Applied force')
    ax.set_ylabel("Free energy G(c)")
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
