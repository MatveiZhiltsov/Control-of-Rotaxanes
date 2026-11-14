from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007,
    "O": 15.999, "S": 32.06, "F": 18.998, "P": 30.974
}

# funcgroups
GROUPS = {
    "ethyl": {
        "ring": list(range(252, 324)),
        "ttf": list(range(28, 46)),
        "dnp": list(range(1, 12)) + [13]
    },
    "propyl": {
        "ring": list(range(258, 330)),
        "ttf": list(range(33, 50)),
        "dnp": list(range(1, 12)) + [13]
    }
}


N_IMAGES = {"ethyl": 3, "propyl": 4}


ETHYL_NEB_ENERGIES = [
    [-458.44823, -458.44763, -458.44591, -458.44324, -458.43984,
     -458.43528, -458.42288, -458.38879, -458.12659, -454.86717],
    [-454.86718, -458.41693, -458.43561, -458.43262, -458.43182,
     -458.43126, -458.42939, -458.42327, -458.37464, -454.84089],
    [-454.84084, -458.42547, -458.42211, -458.41653, -458.40224,
     -458.39215, -458.41818, -458.42549, -458.43056, -458.43970],
]


ENERGIES = {
    "ethyl":  [-454.86717, -454.84089, -458.43970],
    "propyl": [-461.11449, -461.12550, -461.12796, -464.78549]
}


def find_file(variant, number):
    filename = f"{variant}{number:02d}.xyz"
    flat = BASE / filename
    nested = BASE / variant / f"{number:02d}.xyz"
    if flat.exists():
        return flat
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Missing {filename}. Put it beside this script, "
        f"or put {number:02d}.xyz inside the '{variant}' folder."
    )


def read_xyz(path):
    lines = path.read_text().splitlines()
    count = int(lines[0].strip())
    atoms = [line.split() for line in lines[2:2 + count]]
    elements = [atom[0] for atom in atoms]
    xyz = np.array([[float(x) for x in atom[1:4]] for atom in atoms])
    return elements, xyz


def centre(numbers, elements, xyz):
    indices = np.array(numbers) - 1  # XYZ numbers begin at 1; Python begins at 0.
    if min(indices) < 0 or max(indices) >= len(elements):
        raise ValueError(
            f"Atom numbers exceed XYZ atom count ({len(elements)}). "
            "Check whether Avogadro labels are 0-based."
        )
    masses = [MASS[elements[i]] for i in indices]
    return np.average(xyz[indices], axis=0, weights=masses)


def reaction_coordinate(path, groups):
    elements, xyz = read_xyz(path)
    ring = centre(groups["ring"], elements, xyz)
    ttf = centre(groups["ttf"], elements, xyz)
    dnp = centre(groups["dnp"], elements, xyz)
    direction = dnp - ttf
    direction = direction / np.linalg.norm(direction)
    return np.dot(ring - ttf, direction)


plotted = False
for variant in ["ethyl", "propyl"]:
    coordinates = []
    for i in range(1, N_IMAGES[variant] + 1):
        path = find_file(variant, i)
        coordinates.append(reaction_coordinate(path, GROUPS[variant]))

    c = np.array(coordinates)
    c = c - c[0]
    print(f"{variant.title()} displacement (Angstrom): {np.round(c, 3)}")
    if not np.all(np.diff(c) > 0):
        print("  Note: reaction coordinate is not strictly increasing; check image order.")

    np.savetxt(BASE / f"{variant}_coordinates.csv", c,
               delimiter=",", header="displacement_A", comments="")

    values = ENERGIES[variant]
    

    E = (np.array(values, dtype=float) - float(values[0])) * 2625.50
    np.savetxt(BASE / f"{variant}_profile.csv", np.column_stack([c, E]),
               delimiter=",", header="displacement_A,relative_energy_kJmol", comments="")
    plt.plot(c, E, "o-", label=variant.title())
    plotted = True

if plotted:
    plt.xlabel("Ring displacement from image 01 (Angstrom)")
    plt.ylabel("Relative energy (kJ/mol)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / "energy_profiles.png", dpi=200)
    plt.show()
