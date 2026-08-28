# =============================================================================
# ========== Imports ==========

import math
import numpy as np # pyright: ignore[reportMissingImports]
import numba # pyright: ignore[reportMissingImports]

from numba.cuda.random import create_xoroshiro128p_states # pyright: ignore[reportMissingImports]
from numba.cuda.random import xoroshiro128p_uniform_float32 # pyright: ignore[reportMissingImports]
from numba import cuda, uint32, uint64 # pyright: ignore[reportMissingImports]

# ========== Global constants ==========
# ---------- Neighbour tuples ----------
# The following dictionaries provide the relative position
# of a neighbor wrt. a certain lattice site.
# 
# The double list in the dicts is used to handle non-Bravais
# lattice types. E.g.:
# - cubic lattice: all sites have the same nearest neighbours
#   orientation;
# - ? lattice: sites with odd and even indeces have 
#   diffent orientation of nearest neighbours.
# 
# The selection of the 'right' list of neighbours is handled
# below by get_neighbor_displacements_core().

# ===== SIMPLE CUBIC =====
CUBIC_NEIGHBOR_LIST = np.array([
    (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    ),
    (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    )
], dtype=np.int32)

# ===== HEXAGONAL - SINGLE INTERLAYER =====
# 3D hexagonal lattice formed by stacking 2D hexagonal layers,
# where each site has one interlayer neighbour: even and odd sites
# connect alternately upward and downward, respectively.
HEXAGONAL_SINGLE_INTERLAYER_NEIGHBOR_LIST = np.array([
    (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)
    ),
    (
        (1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)
    )
], dtype=np.int32)

# Neighbor list dictionary: given a lattice_type (string), it
# selects the 'right' neighbour list.
NEIGHBOR_LIST_DICT = {
    "cubic"                       : CUBIC_NEIGHBOR_LIST,
    "hexagonal_single_interlayer" : HEXAGONAL_SINGLE_INTERLAYER_NEIGHBOR_LIST
}

# ---------- Lattice selection ----------
lattice_type = "cubic"

# neighbor_list : list of interacting neighbours.
neighbor_list = NEIGHBOR_LIST_DICT[lattice_type]

# Lattice coordination number.
coordination_num = neighbor_list.shape[1]

# ========== Functions ==========
# ---------- Neighbor selection functions ----------
def get_neighbor_displacements_core(
    neighbors,
    i, j, k,
    neighbor_idx
):
    """"
    Return the relative displacement (di, dj, dk)
    of lattice site (i, j, k).

    Parameters
    ----------
    neighbors: np.array[np.int32]
        List of available neighbours depending on the
        lattice type.
    i: int 
        x-index of site in the lattice array.
    j: int
        y-index of site in the lattice array.
    k: int
        z-index of site in the lattice array.
    neighbor_idx: int
        Neighbor index in the neighbour list.

    Return
    ------
    di: int
        Relative x-displacement of the neighbor_idx-th neighbour site.
    dj: int
        Relative y-displacement of the neighbor_idx-th neighbour site.
    dk: int
        Relative z-displacement of the neighbor_idx-th neighbour site.
    """

    parity = (i + j + k) % 2
    di = neighbors[parity, neighbor_idx, 0]
    dj = neighbors[parity, neighbor_idx, 1]
    dk = neighbors[parity, neighbor_idx, 2]
    
    return di, dj, dk

# ---------- Energy functions ----------
def get_pair_energy_v1(type0, type1):
    """
    Return energy of the pair of types i and j.
    """

    # Handle special diagonal
    if type0 == type1:  
        return np.float32(np.inf)

    # Get a unique pair index
    i = np.uint64(type0)
    j = np.uint64(type1)

    # Apply symmetric hash mixing
    a = (227 * 997 * i + 7654321) ^ (887 * 409 * j)
    b = (227 * 997 * j + 7654321) ^ (887 * 409 * i)
    idx = a ^ b
    idx &= 0xFFFFFFFF

    # Mixing function, Wang’s 32-bit hash variant
    idx = (~idx) + (idx << 15)
    idx = idx ^ (idx >> 12)
    idx = idx + (idx << 2)
    idx = idx ^ (idx >> 4)
    idx = idx * uint64(2057)
    idx = idx ^ (idx >> 16)
    idx &= 0xFFFFFFFF  # to 32bit

    # Convert to random float32
    one = np.float32(1.0)
    two = np.float32(2.0)
    x = np.float32(two * np.float32(idx + 1) / np.float32(2 ** 32) - one)  # x e (0.0, 1.0)

    # Edge cases of f32 founding off errors  (avoid infinity when taking logarithm)
    if x <= np.float32(-1.0): x = np.float32(-0.99999994)
    if x >= np.float32(1.0): x = np.float32(0.99999994)

    # S. Winitzki’s (2008), A handy approximation for the error function and its inverse
    # Lecture Notes in Computer Science series, volume 2667
    # https://www.mimirgames.com/articles/programming/approximations-of-the-inverse-error-function
    one_half = np.float32(0.5)
    a = np.float32(0.147)  # 0.1400122886866665
    s = math.copysign(one, x)
    xx = one - x * x
    log_xx = math.log(xx)
    t = two / (math.pi * a) + one_half * log_xx
    inner = t * t - (one / a) * log_xx
    inverf = s * math.sqrt(math.sqrt(inner) - t)  # Eq. (7) in "A handy approximation ..."

    return np.float32(math.sqrt(2.0) * inverf)

# ---------- Function selection ----------
# Neighbours function selection.
h_get_neighbor_displacements = numba.jit(get_neighbor_displacements_core)
d_get_neighbor_displacements = cuda.jit(device=True)(get_neighbor_displacements_core)

# Energy function selection.
get_pair_energy = get_pair_energy_v1    # _v1 = legacy name
h_get_pair_energy = numba.jit(get_pair_energy)
d_get_pair_energy = cuda.jit(device=True)(get_pair_energy)

###################################
##  Host functions: run on CPUs  ##
###################################
@numba.njit
def h_get_particle_energy(
    lattice, 
    xx, yy, zz
):
    """
    Host function: Return energy of the particle 
    located at (xx, yy, zz) in the lattice.
    """

    # Select particle position.
    Lx, Ly, Lz = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy, zz]

    # Loop over neighbours.
    for neighbor_idx in range(coordination_num):
        dx, dy, dz = h_get_neighbor_displacements(
            neighbor_list,
            xx,
            yy,
            zz,
            neighbor_idx
        )

        xx1 = (xx + dx) % Lx
        yy1 = (yy + dy) % Ly
        zz1 = (zz + dz) % Lz
        that_type = lattice[xx1, yy1, zz1]

        energy += h_get_pair_energy(this_type, that_type)

    return energy


@numba.njit(parallel=True)
def h_get_lattice_energy(lattice):
    """
    Host function: Return the energy of the full lattice.
    """
    Lx, Ly, Lz = lattice.shape
    energy = 0.0

    for xx in numba.prange(0, Lx):
        for yy in range(0, Ly):
            for zz in range(0, Lz):
                energy += 0.5 * h_get_particle_energy(
                    lattice, 
                    xx, yy, zz
                )

    return energy


@numba.njit
def run_cpu(
    lattice, 
    displacements, 
    beta, 
    steps
):
    """
    Host function: run full dynamics on CPU.
    """

    Lx, Ly, Lz = lattice.shape
    for _ in range(steps):

        # Pick a random site.
        x0 = np.random.randint(Lx)
        y0 = np.random.randint(Ly)
        z0 = np.random.randint(Lz)

        # Pick random movement direction.
        rnd_int = np.random.randint(coordination_num)
        dx, dy, dz = h_get_neighbor_displacements(
            neighbor_list,
            x0, y0, z0,
            rnd_int
        )
        x1 = (x0 + dx) % Lx
        y1 = (y0 + dy) % Ly
        z1 = (z0 + dz) % Lz

        t0 = lattice[x0, y0, z0]
        t1 = lattice[x1, y1, z1]

        # compute ΔE
        e_old = h_get_particle_energy(
            lattice, 
            x0, y0, z0
        ) + h_get_particle_energy(
            lattice, 
            x1, y1, z1
        )

        lattice[x0, y0, z0], lattice[x1, y1, z1] = t1, t0

        e_new = h_get_particle_energy(
            lattice, 
            x0, y0, z0
        ) + h_get_particle_energy(
            lattice, 
            x1, y1, z1
        )
        delta = e_new - e_old

        # Reject: restore
        if np.random.random() >= math.exp(-beta * delta):
            lattice[x0, y0, z0], lattice[x1, y1, z1] = t0, t1

        # Accept move: and update displacements
        else:  

            this_idx = x0 + y0 * Lx + z0 * Lx * Ly
            that_idx = x1 + y1 * Lx + z1 * Lx * Ly

            this_dx = displacements[3 * this_idx + 0] + dx
            this_dy = displacements[3 * this_idx + 1] + dy
            this_dz = displacements[3 * this_idx + 2] + dz
            that_dx = displacements[3 * that_idx + 0] - dx
            that_dy = displacements[3 * that_idx + 1] - dy
            that_dz = displacements[3 * that_idx + 2] - dz

            displacements[3 * that_idx + 0] = this_dx
            displacements[3 * that_idx + 1] = this_dy
            displacements[3 * that_idx + 2] = this_dz
            displacements[3 * this_idx + 0] = that_dx
            displacements[3 * this_idx + 1] = that_dy
            displacements[3 * this_idx + 2] = that_dz

    return lattice

######################################
##  Device functions: run on GPUs.  ##
######################################
@cuda.jit(device=True)
def d_get_particle_energy(
    lattice, 
    neighbors,
    xx, 
    yy,
    zz
):
    """
    Device function: Return energy of the particle 
    located at (xx, yy, zz) in the lattice.
    """

    # Select particle position.
    Lx, Ly, Lz = lattice.shape
    energy = np.float32(0.0)
    this_type = lattice[xx, yy, zz]

    # Loop over neighbours.
    for neighbor_idx in range(coordination_num):
        dx, dy, dz = d_get_neighbor_displacements(
            neighbors,
            xx,
            yy,
            zz,
            neighbor_idx
        )

        xx1 = (xx + dx) % Lx
        yy1 = (yy + dy) % Ly
        zz1 = (zz + dz) % Lz
        that_type = lattice[xx1, yy1, zz1]

        energy += d_get_pair_energy(this_type, that_type)

    return energy


@cuda.jit(device=True)
def d_update(
    lattice,
    neighbors,
    displacements, 
    beta, 
    tiles, 
    rng_states, 
    step, 
    x, 
    y,
    z
):
    """
    Device function: update state with 1 MC-swap move.
    """

    # ===== Helper variables =====
    Lx, Ly, Lz = lattice.shape
    block_Lx = Lx // tiles[0]
    block_Ly = Ly // tiles[1]

    thread_id = x + y * block_Lx + z * block_Lx * block_Ly
    xx = x * tiles[0]  # My upper-left tile
    yy = y * tiles[1]
    zz = z * tiles[2]

    # ===== Find cell where I'm allowed try neighbour swaps =====
    tile_size = tiles[0] * tiles[1] * tiles[2]
    tile_idx = step % tile_size
    tx = tile_idx % tiles[0]
    ty = (tile_idx // tiles[0]) % tiles[1]
    tz = tile_idx // (tiles[0] * tiles[1])

    xx0 = xx + tx
    yy0 = yy + ty
    zz0 = zz + tz

    # ===== Try swaps on neighbours =====
    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
    rnd_int = int(rnd * float(coordination_num)) % coordination_num
    dx, dy, dz = d_get_neighbor_displacements(
        neighbors,
        xx0, yy0, zz0,
        rnd_int
    )
    xx1 = (xx0 + dx) % Lx
    yy1 = (yy0 + dy) % Ly
    zz1 = (zz0 + dz) % Lz

    # ===== Compute energies =====
    # Old energies.
    energy0_old = d_get_particle_energy(
        lattice,
        neighbors,
        xx0, yy0, zz0
    )
    energy1_old = d_get_particle_energy(
        lattice, 
        neighbors,
        xx1, yy1, zz1
    )

    # Swap
    this_type = lattice[xx0, yy0, zz0]
    that_type = lattice[xx1, yy1, zz1]
    lattice[xx0, yy0, zz0] = that_type
    lattice[xx1, yy1, zz1] = this_type

    # New energies.
    energy0_new = d_get_particle_energy(
        lattice, 
        neighbors,
        xx0, yy0, zz0
    )
    energy1_new = d_get_particle_energy(
        lattice, 
        neighbors,
        xx1, yy1, zz1
    )

    delta = (energy0_new + energy1_new) - (energy0_old + energy1_old)

    # ===== MC-swap move =====
    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
    if rnd < math.exp(-beta * delta):

        # Accept move: record displacements
        this_idx = xx0 + yy0 * Lx + zz0 * Lx * Ly
        dx_this = displacements[3 * this_idx + 0] - dx
        dy_this = displacements[3 * this_idx + 1] - dy
        dz_this = displacements[3 * this_idx + 2] - dz

        that_idx = xx1 + yy1 * Lx + zz1 * Lx * Ly
        dx_that = displacements[3 * that_idx + 0] + dx
        dy_that = displacements[3 * that_idx + 1] + dy
        dz_that = displacements[3 * that_idx + 2] + dz

        displacements[3 * that_idx + 0] = dx_this
        displacements[3 * that_idx + 1] = dy_this
        displacements[3 * that_idx + 2] = dz_this
        displacements[3 * this_idx + 0] = dx_that
        displacements[3 * this_idx + 1] = dy_that
        displacements[3 * this_idx + 2] = dz_that

    else:
        # Reject move: restore lattice
        lattice[xx0, yy0, zz0] = this_type
        lattice[xx1, yy1, zz1] = that_type


@cuda.jit
def kernel_run_simulation(
    lattice,
    neighbors,
    displacements, 
    beta, 
    tiles, 
    rng_states, 
    steps
):
    """
    GPU kernel that runs the simulation.
    Each site attempts to swap with one of its neighbouring sites.
    """
    x, y, z = cuda.grid(3)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0] * tiles[1] * tiles[2]

    for _ in range(steps):
        for tile_step in range(tile_size):
            d_update(
                lattice,
                neighbors,
                displacements, 
                beta, 
                tiles, 
                rng_states, 
                tile_step, 
                x, y, z
            )

            grid.sync()

########################
##  Particle type MC  ##
########################
@cuda.jit(device=True)
def d_update_particle_type_swap(
    lattice, 
    neighbors,
    beta, 
    tiles, 
    rng_states, 
    step, 
    x, y, z
):
    """ Device function: updates state """
    # Helper variables
    num_types = np.uint64(2 ** 32 - 1)
    Lx, Ly, Lz = lattice.shape
    block_Lx = Lx // tiles[0]
    block_Ly = Ly // tiles[1]

    thread_id = x + y * block_Lx + z * block_Lx * block_Ly
    xx = x * tiles[0]  # My upper-left tile
    yy = y * tiles[1]
    zz = z * tiles[2]

    # Find cell where I'm allowed try neighbour swaps
    tile_size = tiles[0] * tiles[1] * tiles[2]
    tile_idx = step % tile_size
    tx = tile_idx % tiles[0]
    ty = (tile_idx // tiles[0]) % tiles[1]
    tz = tile_idx // (tiles[0] * tiles[1])

    xx0 = xx + tx
    yy0 = yy + ty
    zz0 = zz + tz

    energy_old = d_get_particle_energy(
        lattice,
        neighbors,
        xx0, yy0, zz0
    )
    old_type = lattice[xx0, yy0, zz0]
    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)

    new_type = np.uint32(rnd*num_types) % num_types
    lattice[xx0, yy0, zz0] = new_type

    energy_new = d_get_particle_energy(
        lattice, 
        neighbors,
        xx0, yy0, zz0
    )
    delta = energy_new - energy_old
    
    rnd = xoroshiro128p_uniform_float32(rng_states, thread_id)
    if rnd < math.exp(-beta * delta):
        pass  # Accept move
    else:
        lattice[xx0, yy0, zz0] = old_type

@cuda.jit
def kernel_run_type_swaps(
    lattice, 
    neighbors,
    beta, 
    tiles, 
    rng_states, 
    steps
):
    """
    GPU Kernel than run MC simulation where 
    the particle type is swapped.
    """

    x, y, z = cuda.grid(3)
    grid = cuda.cg.this_grid()
    tile_size = tiles[0] * tiles[1] * tiles[2]

    for _ in range(steps):
        for tile_step in range(tile_size):
            d_update_particle_type_swap(
                lattice,
                neighbors,
                beta, 
                tiles, 
                rng_states, 
                tile_step, 
                x, 
                y,
                z
            )

            grid.sync()

# =============================================================================
# ========== Main ==========
def main():
    import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
    import matplotlib as mpl # pyright: ignore[reportMissingModuleSource]
    # mpl.use('Qt5Agg')

    threads_per_block = (4, 4, 4)
    blocks = (12, 12, 12)
    tiles = (4, 4, 4)

    Lx = tiles[0] * blocks[0] * threads_per_block[0]
    Ly = tiles[1] * blocks[1] * threads_per_block[1]
    Lz = tiles[2] * blocks[2] * threads_per_block[2]
    N = Lx * Ly * Lz

    num_of_each_type = 1
    num_types = N // num_of_each_type

    # Setup Lattice
    lattice = np.array(
        [[type] * num_of_each_type for type in range(num_types)], 
        dtype=np.uint32
    ).flatten()
    np.random.shuffle(lattice)
    lattice = lattice.reshape((Lx, Ly, Lz))
    d_lattice = cuda.to_device(lattice)
    displacements = np.zeros(3 * Lx * Ly * Lz, dtype=np.int64)  # 3 values per lattice site
    d_displacements = cuda.to_device(displacements)
    d_neighbor_list = cuda.to_device(neighbor_list) 

    tile_size = tiles[0] * tiles[1] * tiles[2]
    n_threads = N // tile_size
    rng_states = create_xoroshiro128p_states(n_threads, seed=2025)

    # Run simulation with a local particle swap
    beta = 1.6
    steps = 8
    time_blocks = 32
    energies = []
    for _ in range(time_blocks):

        kernel_run_simulation[blocks, threads_per_block](
            d_lattice,
            d_neighbor_list,
            d_displacements, 
            beta, 
            tiles, 
            rng_states, 
            steps
        )

        lattice = d_lattice.copy_to_host()
        energies.append(h_get_lattice_energy(lattice))

    # Run simulation with a swap of types
    for _ in range(time_blocks):
        kernel_run_type_swaps[blocks, threads_per_block](
            d_lattice,
            d_neighbor_list,
            beta, 
            tiles, 
            rng_states, 
            steps
        )
        lattice = d_lattice.copy_to_host()

        energies.append(h_get_lattice_energy(lattice))

    # Plot scaled energies during equilibration
    plt.figure(figsize=(10, 10))
    plt.title(f'Randium. Equilibration, {beta = :0.3f}.')
    scl_enr = 0.5 * np.array(energies) / beta + 1.0
    plt.plot(scl_enr)
    plt.xlabel('Time (steps per particle)')
    plt.ylabel(r'Scaled energy, $1.0+\beta u/2$')
    plt.yscale('log')
    plt.show()

    ## Run short simulation to show displacements
    displacements = np.zeros(3 * Lx * Ly * Lz, dtype=np.int64)
    d_displacements = cuda.to_device(displacements)
    steps = 1024*16
    kernel_run_simulation[blocks, threads_per_block](
        d_lattice,
        d_neighbor_list,
        d_displacements, 
        beta, 
        tiles, 
        rng_states, 
        steps
    )
    displacements = d_displacements.copy_to_host()
    print(displacements)
    dx = displacements[::3].reshape((Lx, Ly, Lz))
    dy = displacements[1::3].reshape((Lx, Ly, Lz))
    dz = displacements[2::3].reshape((Lx, Ly, Lz))
    dr2 = dx**2 + dy**2 + dz**2
    dr = np.sqrt(dr2)
    # msd = sum(sum(dr2)) / N
    # x, y, z = np.meshgrid(np.arange(Lx), np.arange(Ly), np.arange(Lz))

    # print(f'{msd = }')

    # # Plot displacement vectors
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_title(f'Randium. {beta = :0.3f},   {steps = },  {msd =:0.3f}')
    # # Heatmap of dr on the lattice
    # colors = [
    #     "#ffffff",  # white
    #     "#add8e6",  # light blue
    #     "#90ee90",  # light green
    #     "#ffff99",  # light yellow
    #     "#ffcc99",  # light orange
    #     "#ff9999",  # light red
    # ]
    # from matplotlib.colors import LinearSegmentedColormap # pyright: ignore[reportMissingModuleSource]
    # light_rainbow = LinearSegmentedColormap.from_list("light_rainbow", colors, N=256)
    # im = ax.imshow(
    #     dr, origin='lower', interpolation='nearest',
    #     extent=(-0.5, Ly - 0.5, -0.5, Lx - 0.5, Lz ),
    #     cmap=light_rainbow,
    #     vmin=0, vmax=10.0
    # )
    # cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    # cbar.set_label(r'$|\Delta \mathbf{r}|$')

    # mask = ~((dx == 0) & (dy == 0))
    # for xi, yi, dxi, dyi in zip(x[mask].ravel(), y[mask].ravel(),
    #                             dx[mask].ravel(), dy[mask].ravel()):
    #     ax.plot(
    #         [xi, xi + dxi], [yi, yi + dyi],
    #         color='black', linewidth=0.5, alpha=0.5
    #     )
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlim(-0.5, cols + 0.5)
    # ax.set_ylim(-0.5, rows + 0.5)
    # ax.set_aspect('equal', adjustable='box')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()

# =============================================================================