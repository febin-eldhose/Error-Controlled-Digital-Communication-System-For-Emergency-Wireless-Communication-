import numpy as np
import matplotlib.pyplot as plt

# 1. Parity-check matrix
H = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
], dtype=int)

K = 3
N = 6
RATE = K / N


# 2. LDPC encoder
def ldpc_encode(u):
    p1 = u[0] ^ u[1]
    p2 = u[1] ^ u[2]
    p3 = u[0] ^ u[2]

    return np.array(
        [u[0], u[1], u[2], p1, p2, p3],
        dtype=int
    )


# 3. Belief propagation decoder
def ldpc_decode(llr, max_iter=30):
    check_idx, var_idx = np.where(H == 1)
    num_edges = len(check_idx)

    check_edges = [
        np.where(check_idx == i)[0]
        for i in range(H.shape[0])
    ]
    var_edges = [
        np.where(var_idx == j)[0]
        for j in range(N)
    ]

    q = llr[var_idx].astype(float).copy()
    r = np.zeros(num_edges)

    for _ in range(max_iter):

        # Check-node update
        for edges in check_edges:
            for e in edges:
                other = edges[edges != e]
                product = np.prod(np.tanh(q[other] / 2))
                product = np.clip(product, -0.999999, 0.999999)
                r[e] = 2 * np.arctanh(product)

        # Posterior bit reliability
        posterior = llr.copy()

        for j, edges in enumerate(var_edges):
            posterior[j] += np.sum(r[edges])

        decoded = (posterior < 0).astype(int)

        # Check parity constraints
        if np.all((H @ decoded) % 2 == 0):
            break

        # Variable-node update
        for j, edges in enumerate(var_edges):
            for e in edges:
                other = edges[edges != e]
                q[e] = llr[j] + np.sum(r[other])

    return decoded


# 4. AWGN simulation
def simulate(snr_db_values, frames=2000):
    rng = np.random.default_rng(42)

    ber_uncoded = []
    ber_ldpc = []

    for snr_db in snr_db_values:
        eb_n0 = 10 ** (snr_db / 10)
        sigma = np.sqrt(1 / (2 * RATE * eb_n0))

        error_uncoded = 0
        error_ldpc = 0
        total_bits = 0

        for _ in range(frames):
            # Generate information bits
            u = rng.integers(0, 2, K)

            # Encode
            codeword = ldpc_encode(u)

            # BPSK: 0 -> +1, 1 -> -1
            tx = 1 - 2 * codeword

            # Add AWGN
            rx = tx + sigma * rng.standard_normal(N)

            # Uncoded reference
            decoded_uncoded = (rx[:K] < 0).astype(int)

            # Soft information for LDPC decoder
            llr = 2 * rx / (sigma ** 2)
            decoded_ldpc = ldpc_decode(llr)

            # Count information-bit errors
            error_uncoded += np.sum(decoded_uncoded != u)
            error_ldpc += np.sum(decoded_ldpc[:K] != u)
            total_bits += K

        ber_uncoded.append(error_uncoded / total_bits)
        ber_ldpc.append(error_ldpc / total_bits)

    return np.array(ber_uncoded), np.array(ber_ldpc)


# 5. Run simulation
snr_db = np.array([0, 2, 4, 6, 8])

ber_uncoded, ber_ldpc = simulate(snr_db)

print("Eb/N0 (dB) | Uncoded BER | LDPC BER")
print("-----------------------------------")

for snr, b1, b2 in zip(snr_db, ber_uncoded, ber_ldpc):
    print(f"{snr:10.1f} | {b1:11.6f} | {b2:8.6f}")


# 6. Plot BER curves
plt.figure(figsize=(8, 5))

floor = 1 / (2000 * K)

plt.semilogy(
    snr_db,
    np.maximum(ber_uncoded, floor),
    'o-', label='Uncoded BPSK'
)

plt.semilogy(
    snr_db,
    np.maximum(ber_ldpc, floor),
    's-', label='LDPC-coded BPSK'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('LDPC Coding over an AWGN Channel')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
