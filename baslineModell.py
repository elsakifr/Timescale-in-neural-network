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
# Connectivity: balanced E/I baseline
# -------------------
def build_balanced_baseline_connectivity(
    N,
    exc_percent=50,
    w_exc=0.05,
    con_prob=0.2,
):
    NE = int(round(N * exc_percent / 100))
    NI = N - NE

    con_mat = np.zeros((N, N))

    no_con = int(round(con_prob * N))
    w_inh = -w_exc * (NE / NI)

    for j in range(N):
        possible_targets = np.array([i for i in range(N) if i != j])
        tgt_ids = np.random.choice(possible_targets, size=no_con, replace=False)
        con_mat[tgt_ids, j] = 1

    # first NE columns are excitatory
    con_mat[:, :NE] *= w_exc

    # last NI columns are inhibitory
    con_mat[:, NE:] *= w_inh

    return con_mat, NE, NI


# -------------------
# Simulation settings
# -------------------
num_runs = 10
N = 500

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

all_mean_activity = []
all_corr = []

representative_W = None

# -------------------
# Run simulations
# -------------------
for run_id in range(num_runs):
    start_scope()

    current_seed = 10000 + run_id
    seed(current_seed)
    np.random.seed(current_seed)
    defaultclock.dt = dt

    W, NE, NI = build_balanced_baseline_connectivity(
        N=N,
        exc_percent=50,
        w_exc=w_exc,
        con_prob=con_prob,
    )

    if run_id == 0:
        representative_W = W.copy()

        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)

        print(f"Baseline E/I = {NE}/{NI}")
        print(f"Total W sum: {np.sum(W):.10f}")
        print(f"Mean row sum: {np.mean(row_sums):.10f}")
        print(f"Std row sum: {np.std(row_sums):.10f}")
        print(f"Mean col sum: {np.mean(col_sums):.10f}")
        print(f"Std col sum: {np.std(col_sums):.10f}")
        print(f"Min excitatory weight: {np.min(W[:, :NE]):.6f}")
        print(f"Max inhibitory weight: {np.max(W[:, NE:]):.6f}")

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


# -------------------
# Average results
# -------------------
all_mean_activity = np.vstack(all_mean_activity)
all_corr = np.vstack(all_corr)

mean_activity_avg = np.mean(all_mean_activity, axis=0)
std_activity = np.std(all_mean_activity, axis=0)

mean_corr = np.mean(all_corr, axis=0)
std_corr = np.std(all_corr, axis=0)

lags_ms = lags_ms[:1000]

tau_est = estimate_timescale_from_return_to_baseline(
    mean_activity_avg,
    time_ms,
    pulse_start / ms,
    pulse_end / ms,
)

print("\nEstimated baseline timescale:")
if np.isnan(tau_est):
    print("Undefined")
else:
    print(f"{tau_est:.2f} ms")


# -------------------
# Plot 1: mean activity
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(time_ms, mean_activity_avg, label="Mean activity")
plt.fill_between(
    time_ms,
    mean_activity_avg - std_activity,
    mean_activity_avg + std_activity,
    alpha=0.2,
    label="±1 std",
)
plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label="Input pulse")
plt.xlabel("Time (ms)")
plt.ylabel("Mean activity")
plt.title("Baseline balanced E/I network: mean population activity")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 2: autocorrelation
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(lags_ms, mean_corr, label="Mean autocorrelation")
plt.fill_between(
    lags_ms,
    mean_corr - std_corr,
    mean_corr + std_corr,
    alpha=0.2,
    label="±1 std",
)
plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Baseline balanced E/I network: autocorrelation")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 3: connectivity matrix
# -------------------
plt.figure(figsize=(7, 6))
absmax = np.max(np.abs(representative_W))
plt.imshow(representative_W, cmap="bwr", aspect="auto", vmin=-absmax, vmax=absmax)
plt.colorbar(label="Connection strength")
plt.xlabel("Presynaptic neuron j")
plt.ylabel("Postsynaptic neuron i")
plt.title("Baseline balanced E/I connectivity matrix")
plt.tight_layout()

plt.show()