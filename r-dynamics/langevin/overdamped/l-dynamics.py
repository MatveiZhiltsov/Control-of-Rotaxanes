import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
from PIL import Image, ImageDraw
from pathlib import Path

OUT = Path(__file__).resolve().parent


# VARIABLES
a = 1.0
L = 1.0
m = 1.0
y = 2
kBT = 0.04 
dt = 0.01
delta_G = 0.1
t_total = 300
n_paths = 1000
i_save = 50
seed = 11122233


def potential(c, delta_g_val):
    return a * c**2 * (c-L)**2 + delta_g_val * c / L

def force(c, delta_g_val):
    return -(2 * a * c * (c-L) * (2 * c - L) + delta_g_val / L)

def simulation(delta_g_val, n_paths, seed):
    n_steps = int(round(t_total / dt))
    n_saved = n_steps // i_save + 1
    rng = np.random.default_rng(seed)
    times = np.arange(n_saved) * i_save * dt
    samples = np.empty((n_saved, n_paths), dtype=float)
    trace = np.empty(n_steps + 1, dtype=float)
    
    c = np.zeros(n_paths)
    v = rng.normal(0, np.sqrt(kBT/m), size=n_paths)
    samples[0] = c
    trace[0] = c[0]
    noise = np.sqrt(2 * y * kBT / m * dt)

    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        
        # Euler-Maruyama
        dc = v * dt 
        dv = (force(c, delta_g_val) / m - y * v) * dt + noise * z
        c += dc
        v += dv

        trace[step] = c[0]
        if step % i_save == 0:
            samples[step // i_save] = c
                    
    return times, samples, trace

def fmap(solutions):
    c_edges = np.linspace(-0.5, 1.5, 150)
    maps = []
    
    for delta_g_val, (times, positions, trace) in solutions:
        H = np.empty((len(c_edges) - 1, len(times)))
        for col_idx, particles_at_t in enumerate(positions):
            H[:, col_idx], _ = np.histogram(particles_at_t, bins=c_edges, density=True)
            
        maps.append(H)

    vmax = max(np.percentile(matrix, 99.8) for matrix in maps)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True, layout='constrained')
    
    for ax, (delta_g_val, (times, positions, trace)), H in zip(axs, solutions, maps):
        
        im = ax.imshow(H, origin='lower', interpolation='nearest', aspect='auto',
                       extent=(times[0], times[-1], c_edges[0], c_edges[-1]),
                       cmap='magma', vmin=0, vmax=vmax)
                       
        ax.axhline(0, ls=':', lw=1.0, c='white', alpha=0.7)
        ax.axhline(L, ls=':', lw=1.0, c='white', alpha=0.7)
        title = 'Symmetric potential' if delta_g_val == 0 else f'Asymmetric potential ($\\Delta G={delta_g_val}$)'
        ax.set_title(title)
        ax.set_xlabel('Time')
        
    axs[0].set_ylabel('Molecular coordinate $c$')
    fig.colorbar(im, ax=axs, label='Ensemble probability density', shrink=0.85)
    fig.suptitle('Ensemble occupancy: independent inertial Langevin trajectories', fontsize=12)
    
    fig.savefig(OUT / 'langevin_heatmaps.png', dpi=170)
    plt.show()


def fgif(trace, delta_g_val, filename="simulation.gif"):
    skip = 100
    gif_data = trace[::skip]

    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    c_vals = np.linspace(-0.5, 1.5, 300)
    energy = potential(c_vals, delta_g_val)
    ax.plot(c_vals, energy, color='black', linewidth=2, label='Energy Landscape')
    
    dot, = ax.plot([], [], 'ro', markersize=12, label='Molecule')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(min(energy) - 0.05, max(energy) + 0.1)
    ax.set_xlabel("Coordinate, c")
    ax.set_ylabel("Potential energy")
    ax.set_title("Single particle trajectory")
    ax.legend(loc="upper center")

    def upd(frame_index):
        current_c = gif_data[frame_index]
        
        current_energy = potential(current_c, delta_g_val)
        
        dot.set_data([current_c], [current_energy])
        return dot,

    print("Animation is in progress...")
    
    ani = FuncAnimation(fig, upd, frames=len(gif_data), blit=True)
    
    gif_path = OUT / filename
    ani.save(gif_path, writer='pillow', fps=30)
    plt.close(fig)
    print(f"Gif saved to: {gif_path}")


if __name__ == "__main__":
    print("Simulating Symmetric...")
    sol_sym = (0.0, simulation(0.0, n_paths, seed))
    
    print("Simulating Tilted...")
    sol_tilt = (delta_G, simulation(delta_G, n_paths, seed))
    
    print("Trace extraction...")
    tilt_trace = sol_tilt[1][2]
    
    fgif(tilt_trace, delta_G, 'animation.gif')
    
    print("Generating heatmaps...")
    solutions_list = [sol_sym, sol_tilt]
    
    fmap(solutions_list)
    
