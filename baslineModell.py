from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

prefs.codegen.target = "numpy"

# -------------------
# Helper functions
# -------------------
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)


def autocorrelation(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]

    if corr[0] == 0:
        return np.full_like(corr, np.nan)

    return corr / corr[0]


def estimate_timescale_from_return_to_baseline(
    mean_activity,
    time_ms,
    pulse_start_ms,
    pulse_end_ms,
    min_amplitude=0.02,
    return_fraction=0.25,
):
    time_ms = np.asarray(time_ms)
    y = np.asarray(mean_activity)

    baseline_mask = (time_ms >= pulse_start_ms - 200) & (time_ms < pulse_start_ms)
    baseline = np.mean(y[baseline_mask])

    post_mask = time_ms >= pulse_end_ms
    t_post = time_ms[post_mask] - pulse_end_ms
    y_post = y[post_mask]

    response = y_post - baseline

    initial_window = (t_post >= 0) & (t_post <= 50)
    A0 = np.mean(response[initial_window])

    if A0 <= min_amplitude:
        return np.nan

    tail_len = max(20, len(response) // 5)
    final_response = np.mean(response[-tail_len:])

    if abs(final_response) > return_fraction * abs(A0):
        return np.nan

    fit_mask = response > 0.05 * A0
    t_fit = t_post[fit_mask]
    y_fit = response[fit_mask]

    if len(y_fit) < 30:
        return np.nan

    try:
        popt, _ = curve_fit(
            exp_decay,
            t_fit,
            y_fit,
            p0=(A0, 100.0),
            bounds=([0, 1e-6], [np.inf, np.inf]),
            maxfev=10000,
        )
        return popt[1]
    except RuntimeError:
        return np.nan


# -------------------
# Connectivity: Case 2 fixed outdegree, balanced E/I
# -------------------
def build_balanced_fixed_outdegree_connectivity(
    N,
    exc_percent=50,
    w_exc=0.04,
    con_prob=0.2,
):
    NE = int(round(N * exc_percent / 100))
    NI = N - NE

    if NI == 0:
        raise ValueError("NI cannot be zero.")

    con_mat = np.zeros((N, N))

    no_con = int(round(con_prob * N))
    w_inh = -w_exc * (NE / NI)

    for j in range(N):
        possible_targets = np.array([i for i in range(N) if i != j])
        tgt_ids = np.random.choice(possible_targets, size=no_con, replace=False)
        con_mat[tgt_ids, j] = 1

    con_mat[:, :NE] *= w_exc
    con_mat[:, NE:] *= w_inh

    return con_mat, NE, NI


# -------------------
# Simulation settings
# -------------------
num_runs = 10
network_sizes = [50, 100, 200, 500]

duration = 4000 * ms
dt = 0.1 * ms

pulse_start = 500 * ms
pulse_end = 1000 * ms
pulse_amplitude = 0.8
baseline_input = 0.0

tau_baseline = 20 * ms

w_exc = 0.04
con_prob = 0.2
noise_sigma = 0.1

# -------------------
# Storage
# -------------------
results_mean_activity = {}
results_std_activity = {}
results_mean_corr = {}
results_std_corr = {}
results_timescale = {}
results_time = {}
results_lags = {}

# -------------------
# Loop over network sizes
# -------------------
for N in network_sizes:
    all_mean_activity = []
    all_corr = []

    print(f"\nRunning network size N = {N}")

    for run_id in range(num_runs):
        start_scope()

        current_seed = 10000 * run_id + N
        seed(current_seed)
        np.random.seed(current_seed)
        defaultclock.dt = dt

        W, NE, NI = build_balanced_fixed_outdegree_connectivity(
            N=N,
            exc_percent=50,
            w_exc=w_exc,
            con_prob=con_prob,
        )

        if run_id == 0:
            row_sums = np.sum(W, axis=1)
            col_sums = np.sum(W, axis=0)

            print(f"E/I = {NE}/{NI}")
            print(f"Total W sum: {np.sum(W):.10f}")
            print(f"Mean row sum: {np.mean(row_sums):.10f}")
            print(f"Std row sum: {np.std(row_sums):.10f}")
            print(f"Mean col sum: {np.mean(col_sums):.10f}")
            print(f"Std col sum: {np.std(col_sums):.10f}")

        eqs = '''
        dr/dt = (-r + tanh(total_input))/tau_i : 1
        total_input : 1
        tau_i : second
        '''

        G = NeuronGroup(N, eqs, method="euler")
        G.r = "0.05 * rand()"
        G.tau_i = tau_baseline
        G.total_input = baseline_input

        @network_operation(dt=defaultclock.dt)
        def update_input():
            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            if noise_sigma > 0:
                noise = noise_sigma * np.random.randn(N)
            else:
                noise = 0

            G.total_input = input_signal + np.dot(W, G.r) + noise

        M = StateMonitor(G, "r", record=True)

        run(duration)

        mean_activity = np.asarray(np.mean(M.r, axis=0))
        corr = autocorrelation(mean_activity)

        all_mean_activity.append(mean_activity)
        all_corr.append(corr[:1000])

        if run_id == 0:
            time_ms = np.asarray(M.t / ms)
            lags_ms = np.arange(len(corr)) * float(defaultclock.dt / ms)

    all_mean_activity = np.vstack(all_mean_activity)
    all_corr = np.vstack(all_corr)

    results_mean_activity[N] = np.mean(all_mean_activity, axis=0)
    results_std_activity[N] = np.std(all_mean_activity, axis=0)

    results_mean_corr[N] = np.mean(all_corr, axis=0)
    results_std_corr[N] = np.std(all_corr, axis=0)

    results_time[N] = time_ms
    results_lags[N] = lags_ms[:1000]

    results_timescale[N] = estimate_timescale_from_return_to_baseline(
        results_mean_activity[N],
        results_time[N],
        pulse_start / ms,
        pulse_end / ms,
    )

# -------------------
# Plot 1: mean activity
# -------------------
plt.figure(figsize=(10, 6))
for N in network_sizes:
    plt.plot(results_time[N], results_mean_activity[N], label=f"N = {N}")

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label="Input pulse")
plt.xlabel("Time (ms)")
plt.ylabel("Mean activity")
plt.title("Effect of network size: mean population activity")
plt.legend()
plt.tight_layout()

# -------------------
# Plot 2: autocorrelation
# -------------------
plt.figure(figsize=(10, 6))
for N in network_sizes:
    plt.plot(results_lags[N], results_mean_corr[N], label=f"N = {N}")

plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Effect of network size: autocorrelation")
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: timescale vs network size
# -------------------
plt.figure(figsize=(8, 5))

valid_N = []
valid_tau = []

for N in network_sizes:
    tau = results_timescale[N]
    if np.isnan(tau):
        plt.scatter(N, 0, marker="x", s=100)
        plt.text(N, 5, "undefined", ha="center")
    else:
        valid_N.append(N)
        valid_tau.append(tau)

if len(valid_N) > 0:
    plt.plot(valid_N, valid_tau, "o-", label="Estimated timescale")

plt.xlabel("Network size (N)")
plt.ylabel("Estimated timescale (ms)")
plt.title("Effect of network size: estimated post-pulse time constant")
plt.legend()
plt.tight_layout()

plt.show()

print("\nEstimated timescales:")
for N in network_sizes:
    tau = results_timescale[N]
    if np.isnan(tau):
        print(f"N = {N}: undefined")
    else:
        print(f"N = {N}: {tau:.2f} ms")