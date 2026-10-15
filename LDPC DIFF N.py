
import numpy as np
import matplotlib.pyplot as plt


# 1. LDPC encoder and parity-check matrix
def build_ldpc(N):
    K = (N + 1) // 2
    M = N - K

    H = np.zeros((M, N), dtype=int)

    for i in range(M):
        H[i, i % K] = 1
        H[i, (i + 1) % K] = 1
        H[i, K + i] = 1

    return K, H


def ldpc_encode(u, N):
    K = len(u)
    M = N - K

    parity = np.array([
        u[i % K] ^ u[(i + 1) % K]
        for i in range(M)
    ], dtype=int)

    return np.concatenate((u, parity))


# 2. Belief-propagation decoder
def ldpc_decode(llr, H, max_iter=20):
    N = len(llr)

    check_idx, var_idx = np.where(H == 1)
    edges_count = len(check_idx)

    check_edges = [
        np.where(check_idx == i)[0]
        for i in range(H.shape[0])
    ]

    var_edges = [
        np.where(var_idx == j)[0]
        for j in range(N)
    ]

    q = llr[var_idx].astype(float).copy()
    r = np.zeros(edges_count)

    for _ in range(max_iter):

        # Check-node update
        for edges in check_edges:
            for e in edges:
                other = edges[edges != e]

                product = np.prod(np.tanh(q[other] / 2))
                product = np.clip(product, -0.999999, 0.999999)

                r[e] = 2 * np.arctanh(product)

        # Posterior LLR
        posterior = llr.copy()

        for j, edges in enumerate(var_edges):
            posterior[j] += np.sum(r[edges])

        decoded = (posterior < 0).astype(int)

        # Stop if parity checks pass
        if np.all((H @ decoded) % 2 == 0):
            break

        # Variable-node update
        for j, edges in enumerate(var_edges):
            for e in edges:
                other = edges[edges != e]
                q[e] = llr[j] + np.sum(r[other])

    return decoded


# 3. Monte Carlo simulation
def simulate(N, frames=300, snr_db=4, seed=42):
    rng = np.random.default_rng(seed)

    K, H = build_ldpc(N)
    rate = K / N

    eb_n0 = 10 ** (snr_db / 10)

    sigma_uncoded = np.sqrt(1 / (2 * eb_n0))
    sigma_coded = np.sqrt(1 / (2 * rate * eb_n0))

    errors_uncoded = 0
    errors_ldpc = 0
    total_bits = 0

    for _ in range(frames):

        # Random information
        u = rng.integers(0, 2, K)

        # Uncoded transmission
        tx = 1 - 2 * u
        rx = tx + sigma_uncoded * rng.standard_normal(K)

        decoded_uncoded = (rx < 0).astype(int)

        # LDPC-coded transmission
        codeword = ldpc_encode(u, N)
        tx_coded = 1 - 2 * codeword

        rx_coded = (
            tx_coded
            + sigma_coded * rng.standard_normal(N)
        )

        llr = 2 * rx_coded / (sigma_coded ** 2)

        decoded_coded = ldpc_decode(llr, H)

        # Count information-bit errors
        errors_uncoded += np.sum(decoded_uncoded != u)
        errors_ldpc += np.sum(decoded_coded[:K] != u)

        total_bits += K

    ber_uncoded = errors_uncoded / total_bits
    ber_ldpc = errors_ldpc / total_bits

    return ber_uncoded, ber_ldpc, rate


# 4. Test codeword lengths from 3 to 100
N_values = np.arange(3, 101)

frames = 300
snr_db = 4

ber_uncoded_values = []
ber_ldpc_values = []
rates = []

for N in N_values:
    ber_uncoded, ber_ldpc, rate = simulate(
        N,
        frames=frames,
        snr_db=snr_db,
        seed=42 + N
    )

    ber_uncoded_values.append(ber_uncoded)
    ber_ldpc_values.append(ber_ldpc)
    rates.append(rate)

    if N % 10 == 0 or N == 3:
        print(
            f"N={N:3d} | K={(N+1)//2:3d} | "
            f"Rate={rate:.3f} | "
            f"Uncoded BER={ber_uncoded:.6f} | "
            f"LDPC BER={ber_ldpc:.6f}"
        )


# 5. Plot BER against codeword length
plt.figure(figsize=(9, 5))

# Plotting floor for zero-error estimates
floor = 1 / (frames * min((N + 1) // 2 for N in N_values))

plt.semilogy(
    N_values,
    np.maximum(ber_uncoded_values, floor),
    label="Uncoded BPSK",
    marker="o",
    markersize=3
)

plt.semilogy(
    N_values,
    np.maximum(ber_ldpc_values, floor),
    label="LDPC-coded BPSK",
    marker="s",
    markersize=3
)

plt.xlabel("Codeword length N")
plt.ylabel("Bit Error Rate (BER)")
plt.title(f"BER vs Codeword Length at Eb/N0 = {snr_db} dB")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
