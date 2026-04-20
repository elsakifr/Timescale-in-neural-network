from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# -------------------
# Helper functions
# -------------------

def autocorrelation(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr = corr / corr[0]
    return corr

def exp_decay(t, tau):
    return np.exp(-t / tau)

# -------------------
# General settings
# -------------------

num_runs = 20
N = 100

duration = 3000 * ms
dt = 0.1 * ms

w_rec = 0.6

baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms

tau_mean = 20 * ms

# different heterogeneity levels
tau_std_values = [0, 2, 5, 8]

# -------------------
# storage
# -------------------

results_mean_activity = {}
results_mean_corr = {}
timescale_results = {}

representative_t = None
representative_r = {}
representative_tau = {}

lags = None

# -------------------
# simulations
# -------------------

for tau_std_ms in tau_std_values:

    all_mean_activity = []
    all_corr = []

    for run_id in range(num_runs):

        start_scope()

        seed(run_id)
        np.random.seed(run_id)

        defaultclock.dt = dt

        # model equations
        eqs = '''
        dr/dt = (-r + tanh(total_input))/tau_i : 1
        total_input : 1
        tau_i : second
        '''

        G = NeuronGroup(N, eqs, method='euler')

        G.r = '0.05 * rand()'
        G.total_input = baseline_input

        # heterogeneous tau
        if tau_std_ms == 0:

            tau_values = np.ones(N) * float(tau_mean / ms)

        else:

            tau_values = np.random.normal(
                loc=float(tau_mean / ms),
                scale=tau_std_ms,
                size=N
            )

            tau_values = np.clip(tau_values, 1.0, None)

        G.tau_i = tau_values * ms

        # connectivity
        W = np.random.normal(0, w_rec/np.sqrt(N), size=(N, N))
        np.fill_diagonal(W, 0)

        # input update each timestep
        @network_operation(dt=defaultclock.dt)
        def update_input():

            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            G.total_input = input_signal + np.dot(W, G.r)

        # monitor
        M = StateMonitor(G, 'r', record=True)

        run(duration)

        mean_activity = np.mean(M.r, axis=0)

        corr = autocorrelation(mean_activity)

        all_mean_activity.append(mean_activity)
        all_corr.append(corr[:1000])

        # save one example run
        if run_id == 0:

            if representative_t is None:
                representative_t = M.t / ms

            representative_r[tau_std_ms] = np.array(M.r)
            representative_tau[tau_std_ms] = tau_values

            if lags is None:
                lags = np.arange(len(corr)) * float(defaultclock.dt/ms)

    # average across runs
    all_mean_activity = np.array(all_mean_activity)
    all_corr = np.array(all_corr)

    results_mean_activity[tau_std_ms] = np.mean(all_mean_activity, axis=0)
    results_mean_corr[tau_std_ms] = np.mean(all_corr, axis=0)

# -------------------
# compute intrinsic timescale
# -------------------

fit_max_lag = 100

for tau_std_ms in tau_std_values:

    corr_curve = results_mean_corr[tau_std_ms]
    lag_curve = lags[:len(corr_curve)]

    mask = (lag_curve > 0) & (lag_curve <= fit_max_lag)

    x_fit = lag_curve[mask]
    y_fit = corr_curve[mask]

    popt, _ = curve_fit(
        exp_decay,
        x_fit,
        y_fit,
        p0=(20.0,),
        bounds=([1], [1000])
    )

    tau_eff = popt[0]

    timescale_results[tau_std_ms] = tau_eff

    print(
        f"tau std = {tau_std_ms} ms -> intrinsic timescale = {tau_eff:.2f} ms"
    )

# -------------------
# plots
# -------------------

# 1 single neurons
plt.figure(figsize=(10,6))

tau_example = 5

for i in range(5):

    plt.plot(
        representative_t,
        representative_r[tau_example][i],
        label=f'neuron {i}'
    )

plt.axvspan(pulse_start/ms, pulse_end/ms, alpha=0.2)

plt.xlabel('time (ms)')
plt.ylabel('rate')

plt.title(
    f'single neuron activity (tau std = {tau_example} ms)'
)

plt.legend()
plt.tight_layout()

# 2 mean activity
plt.figure(figsize=(10,6))

for tau_std_ms in tau_std_values:

    plt.plot(
        representative_t,
        results_mean_activity[tau_std_ms],
        label=f'tau std = {tau_std_ms}'
    )

plt.axvspan(pulse_start/ms, pulse_end/ms, alpha=0.2)

plt.xlabel('time (ms)')
plt.ylabel('mean activity')

plt.title('mean network activity')

plt.legend()
plt.tight_layout()

# 3 autocorrelation
plt.figure(figsize=(10,6))

for tau_std_ms in tau_std_values:

    plt.plot(
        lags[:1000],
        results_mean_corr[tau_std_ms],
        label=f'tau std = {tau_std_ms}'
    )

plt.xlabel('lag (ms)')
plt.ylabel('autocorrelation')

plt.title('autocorrelation')

plt.legend()
plt.tight_layout()

# 4 tau distribution
plt.figure(figsize=(10,6))

for tau_std_ms in tau_std_values:

    plt.hist(
        representative_tau[tau_std_ms],
        bins=15,
        alpha=0.5,
        label=f'tau std = {tau_std_ms}'
    )

plt.xlabel('tau_i (ms)')
plt.ylabel('count')

plt.title('distribution of tau_i')

plt.legend()
plt.tight_layout()

# 5 intrinsic timescale vs heterogeneity
plt.figure(figsize=(8,5))

x_vals = list(timescale_results.keys())
y_vals = list(timescale_results.values())

plt.plot(x_vals, y_vals, marker='o')

plt.xlabel('tau std (ms)')
plt.ylabel('intrinsic timescale (ms)')

plt.title('intrinsic timescale vs heterogeneity')

plt.tight_layout()

plt.show()