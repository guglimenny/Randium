""" This script conduct an simulation of Randium """

import sys
import toml
import pandas as pd
from numba import cuda
import randium_3d_gpu as rd3
import time

# Print information about GPU
device = cuda.get_current_device()
if cuda.is_available():
    device = cuda.get_current_device()
    print('Device: ', device.name.decode())
else:
    print("WARNING: No CUDA device found")


beta = float(sys.argv[1])
sim_max_time = 10 ** float(sys.argv[2])
num_runs = int(sys.argv[3])

# beta = 1.0
# sim_max_time = 1e2
# num_runs = 4

tic = time.perf_counter()

print(time.ctime())
print(f'{beta = }  {sim_max_time = }  {num_runs = }')

sim_max_time_cpu = 1e4
base = 1.1

data_logs_cpu = []
data_logs_gpu = []

for idx in range(num_runs):
    
    rdm = rd3.Randium_3d_gpu(
        threads_per_block=(4, 4, 4), 
        blocks=(3, 3, 3),
        tiles=(4, 4, 4)
    )

    # Equilibrate with global particle swaps
    # Equilibration criterion (see Randium paper):
    # - u = average energy
    # - beta = inverse temperature
    # - z = coordination number
    # --> expected <u> = - z / 2. *beta
    steps_per_timeblock = 32
    u = 0.0
    steps = 0
    coordination_num = rdm.coordination_num
    energy_coeff_inv = 2. / coordination_num
    beta_inv = 1./ beta     # aka temperature
    energy_target_value = 1. - 0.01     # expected - tollerance

    print(f'  Equilibration ({idx})')
    print('     steps    u    -2/z * u / beta')
    print(time.ctime())

    while -energy_coeff_inv * u * beta_inv < energy_target_value:

        wc = rdm.run_type_swaps(beta=beta, steps=steps_per_timeblock)
        steps += steps_per_timeblock
        u = rdm.energy()
        print('Eqb: ',steps, u, -energy_coeff_inv * u * beta_inv)

    rdm.run_type_swaps(beta=beta, steps=steps)
    u = rdm.energy()
    print('Eqb: ',steps*2, u, -energy_coeff_inv * u * beta_inv)

    # Neighbour particle swaps with CPU implimentation (short time dynamics)
    print(f'  Production, CPU ({idx})')
    print(time.ctime())

    sim_time = 0
    delta_time_float = 1.0
    rdm.reset_data_log(id=idx)

    while sim_time < sim_max_time_cpu:

        rdm.run_cpu(beta=beta, steps=int(delta_time_float))
        print('.', end='', flush=True)
        data_log = rdm.update_data_log()
        sim_time = data_log['time']
        delta_time_float *= base

    data_logs_cpu += rdm.get_data_log()
    print('')

    # Neighbour particle swaps with GPU implimentation  (long time dynamics)
    print(f'  Production, GPU ({idx})')
    print(time.ctime())

    sim_time = 0.0
    delta_time_float = 1.0
    rdm.reset_data_log(id=idx)

    while sim_time < sim_max_time:

        rdm.run(beta=beta, steps=int(delta_time_float))
        print('.', end='', flush=True)
        data_log = rdm.update_data_log()
        sim_time = data_log['time']
        delta_time_float *= base

    data_logs_gpu += rdm.get_data_log()
    print('')

print(f'  Save data to disk')
# Save CPU data to disk
fname = f'Data/cpu_{rdm.Lx}x{rdm.Ly}x{rdm.Lz}beta{beta:.4f}'
toml.dump(dict(**rdm.meta_info(), beta=beta, sim_max_time=sim_max_time_cpu, num_runs=num_runs), open(fname + '.toml', 'w'))
df_all = pd.DataFrame(data_logs_cpu)
df = df_all.groupby('time', as_index=False).mean()
df = df.drop('id', axis=1)
df_std = df_all.groupby('time', as_index=False).std()
df_std = df_std.drop('id', axis=1)
print(df)
df.to_csv(fname + '.csv', index=False, float_format='%.9f')

# Save GPU data to disk
fname = f'Data/gpu_{rdm.Lx}x{rdm.Ly}x{rdm.Lz}beta{beta:.4f}'
toml.dump(dict(**rdm.meta_info(), beta=beta, sim_max_time=sim_max_time, num_runs=num_runs), open(fname + '.toml', 'w'))
df_all = pd.DataFrame(data_logs_gpu)
df_all['time'] = df_all['time'].astype(int)
df = df_all.groupby('time', as_index=False).mean()
df = df.drop('id', axis=1)
df.to_csv(fname + '.csv', index=False, float_format='%.9f')
df_std = df_all.groupby('time', as_index=False).std()
df_std = df_std.drop('id', axis=1)
df_std.to_csv(fname + '_std.csv', index=False, float_format='%.9f')
print(df)

print(time.ctime())

elapsed = time.perf_counter() - tic

print(f'''  Done
Elapsed time: {elapsed:.2f} seconds
            = {elapsed/60:.2f} minutes
            = {elapsed/3600:.2f} hours
            = {elapsed/86400:.2f} days
            = {elapsed/604800:.2f} weeks
            = {elapsed/2629743:.2f} months
''')
