from itertools import product

import numpy as np # pyright: ignore[reportMissingImports]
import numba.cuda # pyright: ignore[reportMissingImports]
from numba.cuda.random import create_xoroshiro128p_states # pyright: ignore[reportMissingImports]

from . import backend

class Randium_2d_gpu:
    def __init__(
            self,
            threads_per_block=(8, 8),
            blocks=(16, 16),
            tiles=(8, 8),
            num_of_each_type=1
    ):

        # ===== Setup GPU params =====
        self.threads_per_block = threads_per_block
        self.blocks = blocks
        self.tiles = tiles
        self.rows = tiles[0] * blocks[0] * threads_per_block[0]
        self.cols = tiles[1] * blocks[1] * threads_per_block[1]

        # ===== Setup lattice =====
        # N = num. of lattice sites = num. of particles
        # N_M = num. of distinc pairs of particle types
        # z = coordination number of the lattices
        self.N = self.rows * self.cols
        self.num_of_each_type = num_of_each_type
        self.num_types = self.N // self.num_of_each_type
        self.N_M = self.num_types * (self.num_types - 1) // 2
        self.z = backend.z

        self.lattice = np.array(
            [[t] * num_of_each_type for t in range(self.num_types)], 
            dtype=np.uint32
        ).flatten()
        np.random.shuffle(self.lattice)
        self.lattice = self.lattice.reshape((self.rows, self.cols))
        self.d_lattice = numba.cuda.to_device(self.lattice)

        self.displacements = np.zeros(2*self.rows*self.cols, dtype=np.int64)
        self.d_displacements = numba.cuda.to_device(self.displacements)

        self.neighbor_list = backend.neighbor_list
        self.d_neighbor_list = numba.cuda.to_device(self.neighbor_list)

        self.sim_time = 0.0

        # Setup random number generator
        tile_size = self.tiles[0] * self.tiles[1]
        n_threads = self.N // tile_size
        self.rng_states = create_xoroshiro128p_states(int(n_threads), seed=2025)

        # Data collected during run executions
        self.steps = []
        self.wallclock_times = []

        # Get ready to record data
        self.reset_data_log()

    def __repr__(self):
        out = f'Randium(threads_per_block=({self.threads_per_block[0]}, {self.threads_per_block[1]}), '
        out += f'blocks=({self.blocks[0]}, {self.blocks[1]}), '
        out += f'tiles=({self.tiles[0]}, {self.tiles[1]}), '
        out += f'num_of_each_type={self.num_of_each_type})'
        return out

    def __str__(self):
        out = self.__repr__() + '\n'
        out += f'  System size: {int(self.cols)} x {int(self.rows)} = {int(self.N)}' '\n'
        out += f'  Number of types: {int(self.num_types)}' '\n'
        out += f'  Unique type pairs: {int(self.N_M)}'
        return out

    def set_lattice(self, lattice):
        self.lattice = lattice
        self.d_lattice = numba.cuda.to_device(lattice)
        self.reset_data_log()

    def get_lattice_site_energies(self):
        output = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            output[i] = backend.h_get_particle_energy(
                self.lattice, 
                i % self.rows, 
                i // self.rows
            )

        return output

    def run(self, beta, steps=1):
        """
        Run the simulation with local swaps on the GPU.
        """
        start = numba.cuda.event()
        end = numba.cuda.event()

        start.record()
        backend.kernel_run_simulation[self.blocks, self.threads_per_block](
            self.d_lattice,
            self.d_neighbor_list,
            self.d_displacements,
            beta,
            self.tiles,
            self.rng_states,
            steps
        )
        end.record()
        end.synchronize()
        wallclock_time = start.elapsed_time(end)

        self.lattice = self.d_lattice.copy_to_host()
        self.displacements = self.d_displacements.copy_to_host()

        self.wallclock_times.append(wallclock_time)
        self.steps.append(steps)

        self.sim_time += float(steps)

        return wallclock_time

    def run_cpu(self, beta, steps=1):
        """
        Run the simulation with local swaps on the CPU.
        """
        self.lattice = backend.run_cpu(
            self.lattice, 
            self.displacements, 
            beta, 
            steps
        )
        self.sim_time += float(steps)/self.N

        return None

    def run_type_swaps(self, beta, steps=1, record_benchmarks=True):
        """
        Run type swaps on the GPU.
        """

        # self.num_types = self.num_types = 2**32-1
        # self.N_M = self.num_types * (self.num_types - 1) // 2
        start = numba.cuda.event()
        end = numba.cuda.event()

        start.record()
        backend.kernel_run_type_swaps[self.blocks, self.threads_per_block](
            self.d_lattice,
            self.d_neighbor_list,
            beta,
            self.tiles,
            self.rng_states,
            steps
        )
        end.record()
        end.synchronize()
        wallclock_time = start.elapsed_time(end)

        self.lattice = self.d_lattice.copy_to_host()

        if record_benchmarks:
            self.wallclock_times.append(wallclock_time)
            self.steps.append(steps)

        self.sim_time += float(steps)

        return wallclock_time

    def energy(self):
        return backend.h_get_lattice_energy(self.lattice) / self.N

    def get_mean_square_displacement(self):
        return np.sum(self.displacements**2) / self.N

    def get_particle_lattice_displacements(self):

        dx = self.displacements[::2].reshape((self.rows, self.cols))
        dy = self.displacements[1::2].reshape((self.rows, self.cols))
        dr2 = dx ** 2 + dy ** 2
        dr = np.sqrt(dr2)
        x, y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))

        return dx, dy, dr, x, y

    def reset_data_log(self, id=0):
        self.sim_time = 0.0
        self.id = int(id)
        self.ref_lat = self.lattice.copy()
        self.displacements = np.zeros(2*self.rows*self.cols, dtype=np.int64)
        self.d_displacements = numba.cuda.to_device(self.displacements)
        self.data_log = []

    def update_data_log(self):
        dx, dy, dr, x, y = self.get_particle_lattice_displacements()
        D = 2  # Spatial dimensions
        msd = float(np.sum(dr**2)/self.N)
        mqd = float(np.sum(dr**4)/self.N)
        ngp = -1.0
        if msd > 0.0:
            ngp = mqd/msd**2/2.0-1.0

        self.data_log.append(dict(
            id = int(self.id),
            time = self.sim_time,
            mean_squared_displacement = msd,
            mean_quartic_displacement = mqd,
            non_gaussian_parameter = ngp,
            energy = self.energy(),
            particle_overlap = float(np.sum(dr<0.5)/ self.N),
            type_overlap = float(np.sum(self.lattice == self.ref_lat) / self.N),
        ))

        return self.data_log[-1]

    def get_data_log(self):

        return self.data_log

    def benchmark(self):

        first_delta_t = self.wallclock_times[0]
        delta_t_avg = np.mean(self.wallclock_times[1:])
        mc_attempts_per_step = self.rows * self.cols
        steps_avg = np.mean(self.steps[1:])

        return dict(
            first_delta_t=float(first_delta_t),  # in ms
            delta_t_avg=float(delta_t_avg),  # in ms
            mc_attempts_per_step=int(mc_attempts_per_step),
            steps_avg=int(steps_avg),
            mc_attempts_per_sec=float(steps_avg * mc_attempts_per_step / (delta_t_avg/1000)),
        )

    def meta_info_simple(self):
        """
        Return a concise dictionary of relevant simulation
        parameters.

        N   = num. of lattice sites = num. of particles
        M   = num. of particle types
        N_m = num. of each type of particle
        N_M = num. of particle type pairs.
        z   = coordination number.
        """

        return dict(
            rows = int(self.rows),
            cols = int(self.cols),
            N    = int(self.N),
            M    = int(self.num_types),
            N_m  = int(self.num_of_each_type),
            N_M  = int(self.N_M),
            z    = int (self.z)
        )

    def meta_info(self):
        """
        Returns the full dictionary of relevant simulation
        parameters.

        N   = num. of lattice sites = num. of particles
        M   = num. of particle types
        N_m = num. of each type of particle
        N_M = num. of particle type pairs.
        z   = coordination number.
        """

        return dict(
            rows = int(self.rows),
            cols = int(self.cols),
            N    = int(self.N),
            M    = int(self.num_types),
            N_m  = int(self.num_of_each_type),
            N_M  = int(self.N_M),
            z    = int (self.z),
            threads_per_block=(
                int(self.threads_per_block[0]), 
                int(self.threads_per_block[1])
            ),
            blocks = (int(self.blocks[0]), int(self.blocks[1])),
            tiles  = (int(self.tiles[0]), int(self.tiles[1])),
            energy = self.energy(),
            **self.benchmark(),
        )

    def get_interaction_matrix(self, x_range, y_range):
        """
        Get the interaction matrix for the given x and y ranges.
        """

        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]

        I = np.array([
            backend.h_get_pair_energy(x, y)
            for x, y in product(range(
                x_range[0], 
                x_range[1]), 
                range(y_range[0], y_range[1]
            ))
        ]).reshape((dx, dy))

        return I
