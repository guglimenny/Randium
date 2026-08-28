import toml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.optimize import curve_fit

# GM stuff
fit_beta_list = []
fit_A_list = []
fit_t0_list = []
fit_alpha_list = []
fit_result_list = []


data_dir = "./raw_data"

cm = mpl.colormaps['jet_r']
def beta_to_color(beta):
    beta_max = 1.9
    return cm(beta/beta_max)


def stretch_exponential(x, A, tau, gamma):
    return A*np.exp(-(x/tau) ** gamma)
p0 = 1.0, 0.5, 0.9
t_fit = np.logspace(-1, 7, 128)

def get_overlap_data(beta):
    fname = f'192x192beta{beta:.4f}'
    data = toml.load(f'data/overlaps_{fname}.toml')
    t = np.array(data['times'])
    Q = np.array(data['Q'])
    Q_dot = np.array(data['Q_dot'])
    t_half = data['t_half']
    return t, Q, Q_dot, t_half

betas = [0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] 

  ##############################################
  ##  Stitch data of cpu and gpu simulations  ##
  ##############################################

plt.figure(figsize=(5, 6))
for idx, beta in enumerate(betas):
    # Load CPU data
    fname = f'192x192beta{beta:.4f}'
    cpu_data = toml.load(data_dir + f'/cpu_{fname}.toml')
    N = cpu_data['N']
    cpu_time_data = pd.read_csv(data_dir + f'/cpu_{fname}.csv')
    # print(f'{N = }')
    t_cpu = np.array(cpu_time_data['time'])
    overlaps_cpu = np.array(cpu_time_data['particle_overlap'])
    Q_cpu = (overlaps_cpu - 1/N)/(1-1/N)
    Q_dot_cpu = np.gradient(Q_cpu, t_cpu)
    # plt.plot(t_cpu, Q_cpu, '+', label=r'$\beta=$' f'{beta:.2f} (cpu)', color=idx_to_color(idx))
    gpu_data = toml.load(data_dir + f'/gpu_{fname}.toml')
    gpu_time_data = pd.read_csv(data_dir + f'/gpu_{fname}.csv')
    t_gpu = np.array(gpu_time_data['time'])
    overlaps_gpu = np.array(gpu_time_data['particle_overlap'])
    Q_gpu = (overlaps_gpu - 1 / N) / (1 - 1 / N)
    Q_dot_gpu = np.gradient(Q_gpu, t_gpu)
    # plt.plot(t_gpu, Q_gpu, 'x', label=r'$\beta=$' f'{beta:.2f} (gpu)', color=idx_to_color(idx))
    mask = (t_gpu > 10_000)
    t = np.array(list(t_cpu) + list(t_gpu[mask]))
    overlaps = np.array(list(overlaps_cpu) + list(overlaps_gpu[mask]))
    Q = np.array(list(Q_cpu) + list(Q_gpu[mask]))
    Q_dot = np.array(list(Q_dot_cpu) + list(Q_dot_gpu[mask]))
    #plt.plot(t, overlaps, '+', label=r'$\beta=$' f'{beta:.2f}', color=idx_to_color(idx))
    plt.plot(t, Q, '-', label=r'$\beta=$' f'{beta:.2f}', color=beta_to_color(beta))
    #plt.plot(t, -Q_dot/Q, '-')
    # plt.plot(t_cpu, -np.gradient(Q_cpu, t_cpu)/Q_cpu, '-')
    # plt.plot(t_gpu, -np.gradient(Q_gpu, t_gpu) / Q_gpu, '-')
    idx_stitch = len(t_cpu)
    # print(f'{t_cpu[-1] = }, {t[idx_stitch-1] = }, {idx_stitch = }')

    # print(beta, min(Q), min(Q) < 0.3)
    t_half = 0.0
    # GM: changed criterion to 0.2
    if min(Q) < 0.2:
        mask = (0.2 < Q) & (Q < 0.8)
        popt, pcov = curve_fit(
            stretch_exponential, 
            t[mask], 
            Q[mask], 
            p0=p0,
            bounds=(
                [.8, 0, 0.],
                [1.2, np.inf, 1.]
            )
        )
        p0 = popt
        A, t0, alpha = popt

        fit_beta_list.append(beta)
        fit_A_list.append(A)
        fit_t0_list.append(t0)
        fit_alpha_list.append(alpha)

        # GM stuff
        # np.savetxt(f"fit_result_beta{beta}.dat", X=popt)

        t_half = popt[1]*np.log(2*popt[0])**(1/popt[2])
        # print(f'{beta}, {t_half = }, {popt =}')
        plt.plot(t_fit, stretch_exponential(t_fit, *popt), 'k--')
        print(f'{beta},{t_half}')

    toml.dump({
        'times': [float(x) for x in t],
        'idx_stitch': idx_stitch,
        't_half': t_half,
        'overlaps': [float(x) for x in overlaps],
        'Q': [float(x) for x in Q],
        'Q_dot': [float(x) for x in Q_dot],
    }, open(f'data/overlaps_{fname}.toml', 'w'))

np.savetxt(
    f"fit_result_all.dat", 
    X=[fit_beta_list, fit_A_list, fit_t0_list, fit_alpha_list]
)

plt.legend(frameon=False, fontsize='8', ncol=1)
plt.xlabel(r'Time, $t$')
plt.ylabel(r'Overlap, $Q$')
#plt.yscale('log')
plt.xscale('log')
plt.show()

  #####################
  ##  Plot overlaps  ##
  #####################


fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(5, 4), sharex=True)

ax = axs[0]
ax.hlines(0.5, 2e-1, 1e9, 'k', linestyles='dashed', color='darkgray')
for idx, beta in enumerate(betas):
    t, Q, Q_dot, t_half = get_overlap_data(beta)
    ax.plot(t, Q, '-', label=f'{beta:.1f}', lw = 2.0, color=beta_to_color(beta))
#ax.legend(frameon=False, fontsize=6, labelspacing=-0.01, loc='lower left', ncol=1)
x_fit_0 = np.logspace(0, 6, 512)
A, gamma, tau = 0.98, 0.5, 1e4
y_fit_0 = stretch_exponential(x_fit_0, A, tau, gamma)
ax.plot(x_fit_0, y_fit_0, 'r--', lw=3.0)
x_fit_1 = np.logspace(-3, 1, 512)
A, gamma, tau = 1.0, 1.0, 0.5  # High T, short timescale
y_fit_1 = stretch_exponential(x_fit_1, A, tau, gamma)
ax.plot(x_fit_1, y_fit_1, 'k--', lw=1.5)
x_fit_2 = np.logspace(-0.5, 2, 512)
y_fit_2 = 1/(2*np.pi*x_fit_2)
ax.plot(x_fit_2, y_fit_2, 'g--', lw=2.0)
ax.set_ylim(0.0, 1.01)
ax.set_xlim(1e-4, 1e9)
ax.set_ylabel(r'Overlap, $Q$')
ax.set_xscale('log')
ax.text(0.015, 0.95, r'(a)', transform=ax.transAxes, ha='left', va='top')
ax.legend(frameon=False, fontsize=6, labelspacing=-0.1, loc='lower left', ncol=2)

ax = axs[1]
ax.hlines(0.5, 8e-4, 1e9, 'k', linestyles='dashed', color='darkgray')
for idx, beta in enumerate(betas):
    t, Q, Q_dot, t_half = get_overlap_data(beta)
    ax.plot(t, 1-Q, '-', label=f'{beta:.2f}', lw = 2.0, color=beta_to_color(beta))
ax.plot(x_fit_0, 1-y_fit_0, 'r--', lw=3.0)
ax.plot(x_fit_1, 1-y_fit_1, 'k--', lw=1.5)
ax.plot(x_fit_2, 1-y_fit_2, 'g--', lw=2.0)
ax.set_ylabel(r'$1-Q$')
ax.set_yscale('log')
ax.set_ylim(1e-3, 1.0)
ax.text(0.015, 0.95, r'(b)', transform=ax.transAxes, ha='left', va='top')
ax.set_xlabel(r'Time, $t$')

plt.savefig('figures/overlap_order_parameter.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/overlap_order_parameter.pdf', dpi=300, bbox_inches='tight')
fig.align_labels()
plt.show()

  ##########################
  ## Find universal curve ##
  ##########################

df = pd.read_csv('data/t_halfs_local_192x192.csv')
betas = df['beta']
t_halfs = df['t_half']
mask = betas > 1.3
betas = betas[mask]
t_halfs = t_halfs[mask]

mins_dict = {
    1.1: 1e0,
    1.2: 1e0,
    1.3: 1e-1,
    1.4: 1e-2,
    1.5: 1e-3,
    1.6: 1e-4,
    1.7: 2e-6,
    1.8: 1e-7,
}

t_save, Q_save = [], []
plt.figure(figsize=(5, 7))
for idx, (beta, t_half) in enumerate(zip(betas, t_halfs)):
    t, Q, *_ = get_overlap_data(beta)
    tt = t/t_half
    plt.plot(tt, 1-Q, 'o', ms=3, label=r'$\beta=$ 'f'{beta:.2f}', color=beta_to_color(beta), alpha=0.3)
    mask = tt > mins_dict[beta]
    tt, Q = tt[mask], Q[mask]
    plt.plot(tt, 1-Q, '+', ms=8, color='black', alpha=1.0)
    t_save += list(tt)
    Q_save += list(Q)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-8, 1e3)
plt.ylim(5e-3, 2)
plt.xticks(np.logspace(-8, 3, 12))
plt.xlabel(r'Scaled Time, $t/t_\text{half}$')
plt.ylabel(r'$1-Q$')
plt.legend(frameon=False, fontsize=12, labelspacing=-0.01, ncol=1)
plt.show()

t_save = np.array(t_save).flatten()
Q_save = np.array(Q_save).flatten()

# Bin the data
num_bins = 36
log_bins = np.logspace(np.log10(min(t_save)), np.log10(max(t_save)), num_bins + 1)
bin_indices = np.digitize(t_save, log_bins)
print(bin_indices)
Q_universal = np.array([
    Q_save[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
    for i in range(1, num_bins+1)
])
t_universal = np.sqrt(log_bins[:-1] * log_bins[1:])  # geometric mean

print(len(t_universal))

plt.figure()
plt.plot(t_universal, 1-Q_universal, marker='o')
plt.xlabel(r'Scaled Time, $t/t_\text{half}$')
plt.ylabel('Q')
plt.xscale('log')
plt.yscale('log')
plt.title('Universal Curve')
plt.grid(True)
plt.show()

# Save to csv file
import pandas as pd
df = pd.DataFrame({
    't': t_universal,
    'Q': Q_universal,
}).to_csv('data/universal_curve.csv', index=False)

#########################
##  t_half scaled plot ##
#########################

fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(5, 4), sharex=True)

ax = axs[0]
for idx, (beta, t_half) in enumerate(zip(betas, t_halfs)):
    t, Q, *_ = get_overlap_data(beta)
    ax.plot(t/t_half, Q, 'o', label=r'$\beta=$' f'{beta:.1f}', lw = 2.0, color=beta_to_color(beta))
x_fit = np.logspace(-8, 3, 512)

A, gamma = 0.98, 0.5
tau = 1.0/(np.log(2*A))**(1/gamma)
ax.plot(x_fit, stretch_exponential(x_fit, A, tau, gamma), 'r--', lw=3.0)

A1, gamma1 = 1.0, 0.5
tau1 = 1.0/(np.log(2*A1))**(1/gamma1)
ax.plot(x_fit, stretch_exponential(x_fit, A1, tau1, gamma1), 'r--', lw=1.0)

ax.plot(t_universal, Q_universal, '--', lw=3.0, color='orange')
ax.text(1e-4, 0.1, r'Universal relaxation', color='orange', fontsize=12, ha='left', va='bottom')
ax.set_ylabel(r'Overlap, $Q$')
ax.set_ylim(0, 1.0)
ax.legend(frameon=False, fontsize=10, labelspacing=-0.01, loc='lower left', ncol=1)
ax.text(0.015, 0.93, r'(a)', transform=ax.transAxes, ha='left', va='top')

ax = axs[1]
for idx, (beta, t_half) in enumerate(zip(betas, t_halfs)):
    t, Q, *_ = get_overlap_data(beta)
    ax.plot(t/t_half, 1-Q, 'o', label=f'{beta:.2f}', lw = 2.0, color=beta_to_color(beta))
ax.plot(x_fit, 1-stretch_exponential(x_fit, A, tau, gamma), 'r--', lw=3.0)
#ax.plot(x_fit, 1-stretch_exponential(x_fit, A1, tau1, gamma1), 'r--', lw=1.0)
ax.plot(t_universal, 1-Q_universal, '--', lw=3.0, color='orange')
ax.text(0.015, 0.93, r'(b)', transform=ax.transAxes, ha='left', va='top')
ax.set_yscale('log')
ax.set_ylim(5e-3, 1.0)
ax.set_ylabel(r'$1-Q$')
ax.set_xlabel(r'Scaled Time, $t/\tau$')

plt.xscale('log')
plt.xlim(1e-8, 1e3)
fig.align_labels()
plt.savefig('figures/overlap_order_scaled.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/overlap_order_scaled.pdf', dpi=300, bbox_inches='tight')
plt.show()

