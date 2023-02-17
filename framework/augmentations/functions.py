# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
# from https://github.com/braindecode

from numbers import Real

import numpy as np
from scipy.interpolate import Rbf
from sklearn.utils import check_random_state
import torch
from torch.fft import fft, ifft
from torch.nn.functional import pad, one_hot
from mne.filter import notch_filter


def identity(X, y):
    return X, y


def time_reverse(X):
    return torch.flip(X, [-1])


def sign_flip(X):
    return -X


def _new_random_fft_phase_odd(batch_size, c, n, device, random_state):
    rng = torch.rand(batch_size, c, (n - 1) // 2, device=device)
    random_phase = 2j * np.pi * rng
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        -torch.flip(random_phase, [-1])
    ], dim=-1)


def _new_random_fft_phase_even(batch_size, c, n, device, random_state):
    rng = torch.rand(batch_size, c, n // 2 - 1, device=device)
    random_phase = 2j * np.pi * rng
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        torch.zeros((batch_size, c, 1), device=device),
        -torch.flip(random_phase, [-1])
    ], dim=-1)


_new_random_fft_phase = {
    0: _new_random_fft_phase_even,
    1: _new_random_fft_phase_odd
}


def ft_surrogate(
    X,
    phase_noise_magnitude,
    channel_indep,
    random_state=None
):
    assert isinstance(
        phase_noise_magnitude,
        (Real, torch.FloatTensor, torch.cuda.FloatTensor)
    ) and 0 <= phase_noise_magnitude <= 1, (
        f"eps must be a float beween 0 and 1. Got {phase_noise_magnitude}.")

    f = fft(X.double(), dim=-1)
    device = X.device

    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        f.shape[0],
        f.shape[-2] if channel_indep else 1,
        n,
        device=device,
        random_state=random_state
    )
    if not channel_indep:
        random_phase = torch.tile(random_phase, (1, f.shape[-2], 1))
    if isinstance(phase_noise_magnitude, torch.Tensor):
        phase_noise_magnitude = phase_noise_magnitude.to(device)
    f_shifted = f * torch.exp(phase_noise_magnitude * random_phase)
    shifted = ifft(f_shifted, dim=-1)
    transformed_X = shifted.real.float()

    return transformed_X


def _pick_channels_randomly(X, p_pick, random_state):
    rng = check_random_state(random_state)
    n_channels, _ = X.shape
    # allows to use the same RNG
    unif_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    # equivalent to a 0s and 1s mask
    return torch.sigmoid(1000*(unif_samples - p_pick))


def channels_dropout(X, p_drop, random_state=None):
    mask = _pick_channels_randomly(X, p_drop, random_state=random_state)
    return X * mask.unsqueeze(-1)


def column_dropout(table, p_drop, indices, random_state=None):
    for i in range(table.size(-1)):
        if p_drop > np.random.uniform() or i in indices:
            table[i] *= 0.0
    return table

def _make_permutation_matrix(X, mask, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        channels_permutation = np.arange(n_channels)
        channels_permutation[channels_to_shuffle] = rng.permutation(
            channels_to_shuffle
        )
        channels_permutation = torch.as_tensor(
            channels_permutation, dtype=torch.int64, device=X.device
        )
        batch_permutations[b, ...] = one_hot(channels_permutation)
    return batch_permutations


def channels_shuffle(X, y, p_shuffle, random_state=None):
    if p_shuffle == 0:
        return X, y
    mask = _pick_channels_randomly(X, 1 - p_shuffle, random_state)
    batch_permutations = _make_permutation_matrix(X, mask, random_state)
    return torch.matmul(batch_permutations, X), y


def gaussian_noise(X, std, random_state=None):
    rng = check_random_state(random_state)
    if isinstance(std, torch.Tensor):
        std = std.to(X.device)
    noise = torch.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        ),
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X


def channels_permute(X, y, permutation):
    return X[..., permutation, :], y


def smooth_time_mask(X, y, mask_start_per_sample, mask_len_samples):
    batch_size, n_channels, seq_len = X.shape
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 / seq_len
    mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
            torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))
            ).float().to(X.device)
    return X * mask, y


def bandstop_filter(X, sfreq, bandwidth, freqs_to_notch):
    if bandwidth == 0:
        return X
    transformed_X = X.clone()
    for c, (sample, notched_freq) in enumerate(
            zip(transformed_X, freqs_to_notch)):
        sample = sample.cpu().numpy().astype(np.float64)
        transformed_X[c] = torch.as_tensor(notch_filter(
            sample,
            Fs=sfreq,
            freqs=notched_freq,
            method='fir',
            notch_widths=bandwidth,
            verbose=False
        ))
    return transformed_X


def _analytic_transform(x):
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    N = x.shape[-1]
    f = fft(x, N, dim=-1)
    h = torch.zeros_like(f)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    return ifft(f * h, dim=-1)


def _nextpow2(n):
    return int(np.ceil(np.log2(np.abs(n))))


def _frequency_shift(X, fs, f_shift):
    # Pad the signal with zeros to prevent the FFT invoked by the transform
    # from slowing down the computation:
    n_channels, N_orig = X.shape[-2:]
    N_padded = 2 ** _nextpow2(N_orig)
    t = torch.arange(N_padded, device=X.device) / fs
    padded = pad(X, (0, N_padded - N_orig))
    analytical = _analytic_transform(padded)
    if isinstance(f_shift, (float, int, np.ndarray, list)):
        f_shift = torch.as_tensor(f_shift).float()
    reshaped_f_shift = f_shift.repeat(
        N_padded, n_channels, 1).T
    shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
    return shifted[..., :N_orig].real.float()


def frequency_shift(X, delta_freq, sfreq):
    transformed_X = _frequency_shift(
        X=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    return transformed_X


def _torch_normalize_vectors(rr):
    norm = torch.linalg.norm(rr, axis=1, keepdim=True)
    mask = (norm > 0)
    norm[~mask] = 1  # in case norm is zero, divide by 1
    new_rr = rr / norm
    return new_rr


def _torch_legval(x, c, tensor=True):
    c = torch.as_tensor(c)
    c = c.double()
    if isinstance(x, (tuple, list)):
        x = torch.as_tensor(x)
    if isinstance(x, torch.Tensor) and tensor:
        c = c.view(c.shape + (1,)*x.ndim)

    c = c.to(x.device)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x


def _torch_calc_g(cosang, stiffness=4, n_legendre_terms=50):
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi)
               for n in range(1, n_legendre_terms + 1)]
    return _torch_legval(cosang, [0] + factors)


def _torch_make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    pos_from = pos_from.clone()
    pos_to = pos_to.clone()
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # normalize sensor positions to sphere
    pos_from = _torch_normalize_vectors(pos_from)
    pos_to = _torch_normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = torch.matmul(pos_from, pos_from.T)
    cosang_to_from = torch.matmul(pos_to, pos_from.T)
    G_from = _torch_calc_g(cosang_from)
    G_to_from = _torch_calc_g(cosang_to_from)
    assert G_from.shape == (n_from, n_from)
    assert G_to_from.shape == (n_to, n_from)

    if alpha is not None:
        G_from.flatten()[::len(G_from) + 1] += alpha

    device = G_from.device
    C = torch.vstack([
            torch.hstack([G_from, torch.ones((n_from, 1), device=device)]),
            torch.hstack([
                torch.ones((1, n_from), device=device),
                torch.as_tensor([[0]], device=device)])
        ])

    try:
        C_inv = torch.linalg.inv(C)
    except torch._C._LinAlgError:
        # There is a stability issue with pinv since torch v1.8.0
        # see https://github.com/pytorch/pytorch/issues/75494
        C_inv = torch.linalg.pinv(C.cpu()).to(device)

    interpolation = torch.hstack([
        G_to_from,
        torch.ones((n_to, 1), device=device)
    ]).matmul(C_inv[:, :-1])
    assert interpolation.shape == (n_to, n_from)
    return interpolation


def _rotate_signals(X, rotations, sensors_positions_matrix, spherical=True):
    sensors_positions_matrix = sensors_positions_matrix.to(X.device)
    rot_sensors_matrices = [
        rotation.matmul(sensors_positions_matrix) for rotation in rotations
    ]
    if spherical:
        interpolation_matrix = torch.stack(
            [torch.as_tensor(
                _torch_make_interpolation_matrix(
                    sensors_positions_matrix.T, rot_sensors_matrix.T
                ), device=X.device
            ).float() for rot_sensors_matrix in rot_sensors_matrices]
        )
        return torch.matmul(interpolation_matrix, X)
    else:
        transformed_X = X.clone()
        sensors_positions = list(sensors_positions_matrix)
        for s, rot_sensors_matrix in enumerate(rot_sensors_matrices):
            rot_sensors_positions = list(rot_sensors_matrix.T)
            for time in range(X.shape[-1]):
                interpolator_t = Rbf(*sensors_positions, X[s, :, time])
                transformed_X[s, :, time] = torch.from_numpy(
                    interpolator_t(*rot_sensors_positions)
                )
        return transformed_X


def _make_rotation_matrix(axis, angle, degrees=True):
    assert axis in ['x', 'y', 'z'], "axis should be either x, y or z."

    if isinstance(angle, (float, int, np.ndarray, list)):
        angle = torch.as_tensor(angle)

    if degrees:
        angle = angle * np.pi / 180

    device = angle.device
    zero = torch.zeros(1, device=device)
    rot = torch.stack([
        torch.as_tensor([1, 0, 0], device=device),
        torch.hstack([zero, torch.cos(angle), -torch.sin(angle)]),
        torch.hstack([zero, torch.sin(angle), torch.cos(angle)]),
    ])
    if axis == "x":
        return rot
    elif axis == "y":
        rot = rot[[2, 0, 1], :]
        return rot[:, [2, 0, 1]]
    else:
        rot = rot[[1, 2, 0], :]
        return rot[:, [1, 2, 0]]


def sensors_rotation(X, y, sensors_positions_matrix, axis, angles,
                     spherical_splines):
    rots = [
        _make_rotation_matrix(axis, angle, degrees=True)
        for angle in angles
    ]
    rotated_X = _rotate_signals(
        X, rots, sensors_positions_matrix, spherical_splines
    )
    return rotated_X, y


def mixup(X, y, lam, idx_perm):
    device = X.device
    batch_size, n_channels, n_times = X.shape

    X_mix = torch.zeros((batch_size, n_channels, n_times)).to(device)
    y_a = torch.arange(batch_size).to(device)
    y_b = torch.arange(batch_size).to(device)

    for idx in range(batch_size):
        X_mix[idx] = lam[idx] * X[idx] \
            + (1 - lam[idx]) * X[idx_perm[idx]]
        y_a[idx] = y[idx]
        y_b[idx] = y[idx_perm[idx]]

    return X_mix, (y_a, y_b, lam)
