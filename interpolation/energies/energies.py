import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent


ETHYL_NEB_ENERGIES = [
    [-458.44823, -458.44763, -458.44591, -458.44324, -458.43984,
     -458.43528, -458.42288, -458.38879, -458.12659, -454.86717],
    [-454.86718, -458.41693, -458.43561, -458.43262, -458.43182,
     -458.43126, -458.42939, -458.42327, -458.37464, -454.84089],
    [-454.84084, -458.42547, -458.42211, -458.41653, -458.40224,
     -458.39215, -458.41818, -458.42549, -458.43056, -458.43970],
]

PROPYL_NEB_ENERGIES = [
    [-464.77977, -464.77890, -464.77751, -464.77280, -464.77260,
     -464.77092, -464.76847, -464.76140, -464.73280, -461.11449],
    [-461.11447, -464.71307, -464.75575, -464.75722, -464.75513,
     -464.75499, -464.75467, -464.75348, -464.71946, -461.12550],
    None,  #unreasonable energies
    [-461.12796, -464.75203, -464.66715, -464.67153, -464.67206,
     -464.66121, -464.67467, -464.69282, -464.78335, -464.78549],
]

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for ax, (name, segments) in zip(
    axes, [("Ethyl", ETHYL_NEB_ENERGIES), ("Propyl", PROPYL_NEB_ENERGIES)]
):
    initial_energy = segments[0][0]
    for i, energies in enumerate(segments):
        if energies is None:
            start = (segments[i - 1][-1] - initial_energy) * 2625.50
            end = (segments[i + 1][0] - initial_energy) * 2625.50
            ax.plot([i, i + 1], [start, end], color="0.55", ls="--", lw=1.5)
            continue

        x = i + np.linspace(0, 1, len(energies))
        y = (np.array(energies) - initial_energy) * 2625.50
        ax.plot(x, y, "-", lw=2, label=f"Segment {i + 1}")

    ax.set_title(f"{name} rotaxane — all available NEB image energies")
    ax.set_ylabel("Relative energy (kJ/mol)")
    ax.set_yscale("symlog", linthresh=50)
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=9, loc="upper left")

axes[-1].set_xlabel("Consecutive NEB segment progress (each segment = 1)")
axes[-1].set_xlim(0, 4)
axes[-1].set_xticks([0, 1, 2, 3, 4])
fig.tight_layout()
fig.savefig(BASE / "energy_profiles.png", dpi=200)
plt.show()
