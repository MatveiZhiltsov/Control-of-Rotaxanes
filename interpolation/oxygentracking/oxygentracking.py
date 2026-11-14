from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
HARTREE_TO_KJMOL = 2625.50
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "S": 32.06, "F": 18.998, "P": 30.974}

GROUPS = {
    "ethyl": {
        "ring": list(range(252, 324)),
        "ttf": list(range(28, 46)),
        "dnp": list(range(1, 12)) + [13],
        "oxygen": [18, 36],
    },
    "propyl": {
        "ring": list(range(258, 330)),
        "ttf": list(range(33, 50)),
        "dnp": list(range(1, 12)) + [13],
        "oxygen": [21, 42],
    },
}


ENERGIES = {
    "ethyl": {
        1: [-458.44823, -458.44763, -458.44591, -458.44324, -458.43984,
            -458.43528, -458.42288, -458.38879, -458.12659, -454.86717],
        2: [-454.86718, -458.41693, -458.43561, -458.43262, -458.43182,
            -458.43126, -458.42939, -458.42327, -458.37464, -454.84089],
        3: [-454.84084, -458.42547, -458.42211, -458.41653, -458.40224,
            -458.39215, -458.41818, -458.42549, -458.43056, -458.43970],
    },
    "propyl": {
        1: [-464.77977, -464.77890, -464.77751, -464.77280, -464.77260,
            -464.77092, -464.76847, -464.76140, -464.73280, -461.11449],
        2: [-461.11447, -464.71307, -464.75575, -464.75722, -464.75513,
            -464.75499, -464.75467, -464.75348, -464.71946, -461.12550],
        # Propyl 03 excluded, as agreed.
        4: [-461.12796, -464.75203, -464.66715, -464.67153, -464.67206,
            -464.66121, -464.67467, -464.69282, -464.78335, -464.78549],
    },
}


SKIP = {("propyl", 2, 9), ("propyl", 4, 0)}


def read_xyz_trajectory(path):
    """Read the ten repeated XYZ blocks in ethyl01.xyz-style files."""
    lines = path.read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        atoms = [line.split() for line in lines[i + 2:i + 2 + n]]
        elements = [a[0].capitalize() for a in atoms]
        xyz = np.array([[float(v) for v in a[1:4]] for a in atoms])
        frames.append((elements, xyz))
        i += n + 2
    return frames


def centre(atom_numbers, elements, xyz):
    ids = np.array(atom_numbers) - 1
    weights = [MASS[elements[i]] for i in ids]
    return np.average(xyz[ids], weights=weights, axis=0)


def make_data(variant):
    groups = GROUPS[variant]
    reference_energy = ENERGIES[variant][1][0]
    first_ring_position = None
    rows = []

    for segment, energies in ENERGIES[variant].items():
        # Files are named exactly ethyl01.xyz / propyl04.xyz, etc.
        frames = read_xyz_trajectory(BASE / f"{variant}{segment:02d}.xyz")
        if len(frames) != 10:
            raise ValueError(f"{variant}{segment:02d}.xyz must contain 10 NEB images")

        for image, ((elements, xyz), energy) in enumerate(zip(frames, energies)):
            if (variant, segment, image) in SKIP:
                continue

            ttf = centre(groups["ttf"], elements, xyz)
            dnp = centre(groups["dnp"], elements, xyz)
            ring = centre(groups["ring"], elements, xyz)
            axis = (dnp - ttf) / np.linalg.norm(dnp - ttf)
            ring_position = float(np.dot(ring - ttf, axis))
            if first_ring_position is None:
                first_ring_position = ring_position

            ring_xyz = xyz[np.array(groups["ring"]) - 1]
            row = {
                "segment": segment,
                "image": image,
                "ring_displacement_A": ring_position - first_ring_position,
                "energy_Ha": energy,
                "relative_energy_kJmol": (energy - reference_energy) * HARTREE_TO_KJMOL,
            }
            for atom_number in groups["oxygen"]:
                oxygen_xyz = xyz[atom_number - 1]
                oxygen_position = float(np.dot(oxygen_xyz - ttf, axis))
                row[f"O{atom_number}_position_A"] = oxygen_position - first_ring_position
                row[f"O{atom_number}_minus_ring_A"] = oxygen_position - ring_position
                row[f"O{atom_number}_min_contact_A"] = float(
                    np.linalg.norm(ring_xyz - oxygen_xyz, axis=1).min()
                )
            rows.append(row)
    return rows


def plot(variant, rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    for segment in ENERGIES[variant]:
        points = [r for r in rows if r["segment"] == segment]
        ax.plot([r["ring_displacement_A"] for r in points],
                [r["relative_energy_kJmol"] for r in points],
                "o-", ms=4, label=f"NEB segment {segment}")

    for atom_number, color in zip(GROUPS[variant]["oxygen"], ("tab:orange", "tab:green")):
        positions = np.array([r[f"O{atom_number}_position_A"] for r in rows])
        ax.axvline(np.median(positions), ls="--", color=color,
                   label=f"O atom {atom_number}")
        ax.axvspan(positions.min(), positions.max(), color=color, alpha=0.1)

    ax.set_xlabel("Macrocycle displacement from first image (Å)")
    ax.set_ylabel("Relative NEB energy (kJ/mol)")
    ax.set_yscale("symlog", linthresh=30)
    ax.set_title(f"{variant.title()} — oxygen positions and NEB energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / f"{variant}_oxygen_energy.png", dpi=200)
    plt.close(fig)


def main():
    for variant in ("ethyl", "propyl"):
        rows = make_data(variant)
        with (BASE / f"{variant}_oxygen_energy.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        plot(variant, rows)


if __name__ == "__main__":
    main()

