
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


# 3. Belief-propagation LDPC decoder
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

                product = np.prod(
                    np.tanh(q[other] / 2)
                )

                product = np.clip(
                    product, -0.999999, 0.999999
                )

                r[e] = 2 * np.arctanh(product)

        # Calculate posterior LLRs
        posterior = llr.copy()

        for j, edges in enumerate(var_edges):
            posterior[j] += np.sum(r[edges])

        # Make hard decisions
        decoded = (posterior < 0).astype(int)

        # Stop if all parity checks pass
        if np.all((H @ decoded) % 2 == 0):
            break

        # Variable-node update
        for j, edges in enumerate(var_edges):
            for e in edges:
                other = edges[edges != e]
                q[e] = llr[j] + np.sum(r[other])

    return decoded


# 4. Monte Carlo simulation
def monte_carlo(snr_db_values, frames=2000):

    rng = np.random.default_rng(42)

    ber_uncoded = []
    ber_ldpc = []

    for snr_db in snr_db_values:

        # Convert Eb/N0 from dB to linear scale
        eb_n0 = 10 ** (snr_db / 10)

        # Separate noise levels for equal Eb/N0
        sigma_uncoded = np.sqrt(1 / (2 * eb_n0))
        sigma_coded = np.sqrt(
            1 / (2 * RATE * eb_n0)
        )

        errors_uncoded = 0
        errors_ldpc = 0
        total_bits = 0

        # Monte Carlo trials
        for trial in range(frames):

            # Generate random information bits
            u = rng.integers(0, 2, K)

            # ----- Uncoded transmission -----
            tx_uncoded = 1 - 2 * u

            rx_uncoded = (
                tx_uncoded
                + sigma_uncoded
                * rng.standard_normal(K)
            )

            decoded_uncoded = (
                rx_uncoded < 0
            ).astype(int)

            # ----- LDPC-coded transmission -----
            codeword = ldpc_encode(u)

            # BPSK: 0 -> +1, 1 -> -1
            tx_coded = 1 - 2 * codeword

            # Add AWGN
            rx_coded = (
                tx_coded
                + sigma_coded
                * rng.standard_normal(N)
            )

            # Soft information for BP decoder
            llr = 2 * rx_coded / (sigma_coded ** 2)

            decoded_coded = ldpc_decode(llr)

            # Count information-bit errors
            errors_uncoded += np.sum(
                decoded_uncoded != u
            )

            errors_ldpc += np.sum(
                decoded_coded[:K] != u
            )

            total_bits += K

        # Estimate BER after all trials
        ber_uncoded.append(errors_uncoded / total_bits)
        ber_ldpc.append(errors_ldpc / total_bits)

        print(
            f"Eb/N0 = {snr_db:2} dB | "
            f"Uncoded BER = {ber_uncoded[-1]:.6f} | "
            f"LDPC BER = {ber_ldpc[-1]:.6f} | "
            f"Trials = {frames}"
        )

    return np.array(ber_uncoded), np.array(ber_ldpc)


# 5. Run the Monte Carlo experiment
snr_db = np.array([0, 2, 4, 6, 8])

ber_uncoded, ber_ldpc = monte_carlo(
    snr_db,
    frames=2000
)


# 6. Plot BER curves
plt.figure(figsize=(8, 5))

# Plotting floor only handles zero BER values on log scale
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
plt.title('Monte Carlo Simulation of LDPC over AWGN')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()
