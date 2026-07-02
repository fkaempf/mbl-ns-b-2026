"""Central-complex imaging helpers (lab-standard), used verbatim so the FC2 bump and
normalisation match the rest of the pipeline: population-vector average (`get_pva`),
signal normalisation incl. `maxmin` (`calculate_normalized_signal`), protocerebral-bridge
de-interdigitation (`interdigitate_bridge`), and wrap-line removal (`remove_wrap_lines`).
"""

import numpy as np


def calculate_normalized_signal(signal, method="dF/F0", f0_frac=0.05, fmax_frac=0.95):
    if method == "dF/F0":
        signal_ar = signal.to_numpy()
        f0 = np.nanmean(np.sort(signal_ar)[: int(len(signal_ar) * f0_frac)])
        norm_signal = (signal_ar - f0) / f0

    elif method == "zscore":
        norm_signal = (signal - signal.mean()) / signal.std()

    elif method == "maxmin":
        sorted_signal = signal.sort_values(ascending=True, ignore_index=True)
        f0 = sorted_signal[: int(len(signal) * f0_frac)].mean()
        fmax = sorted_signal[int(len(signal) * fmax_frac) :].mean()
        norm_signal = (signal - f0) / (fmax - f0)

    else:
        print("Normalization method not recognized")
        norm_signal = None

    return norm_signal


def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def get_pva(data):
    """
    Gets phase and amplitude using population vector average method
    :returns theta (degrees) and amplitude

    """
    data = data.copy()
    step = (2 * np.pi) / np.shape(data)[1]
    angles = np.arange(0, 2 * np.pi, step) + step / 2.0
    # subtracts min from each row to remove negative dF/F (used as weights)
    data = (data.T - np.nanmin(data, axis=1)).T
    data = (data.T / np.nansum(data, axis=1)).T
    x, y = polar2cart(data, angles)
    r, theta = cart2polar(np.nansum(x, axis=1), np.nansum(y, axis=1))
    theta[theta < 0] = theta[theta < 0] + 2 * np.pi
    theta *= 180 / np.pi
    return theta, r


def interdigitate_bridge(data, cols_to_extract_pb, append=False):
    data_mat = data[cols_to_extract_pb[1:-1]].transpose().to_numpy()

    data_mat_interdigit = np.zeros(np.shape(data_mat))

    k = 1
    l = 8
    for i in range(0, 16):
        if (i % 2) == 0:
            data_mat_interdigit[i] = data_mat[i + l]
            l -= 1
        else:
            data_mat_interdigit[i] = data_mat[i - k]
            k += 1

    return data_mat_interdigit


def remove_wrap_lines(trace, trsh=180):
    new_trace = trace

    difference = np.diff(trace, prepend=0)

    idx = abs(difference) > trsh
    new_trace[idx] = np.nan

    return new_trace
