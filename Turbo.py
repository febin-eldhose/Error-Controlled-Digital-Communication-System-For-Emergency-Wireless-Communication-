import numpy as np
import matplotlib.pyplot as plt

# ---------------- RSC ENCODER ----------------
def encode(x):
    s1 = s2 = 0
    p = []
    for b in x:
        p.append(b ^ s1 ^ s2)
        s2 = s1
        s1 = b
    return np.array(p)

# ---------------- TRANSMITTER ----------------
x = np.array([1, 0, 1, 1])
p1 = encode(x)

order = [0, 2, 1, 3]
x2 = x[order]
p2 = encode(x2)

print("Input       :", x)
print("Parity 1    :", p1)
print("Interleaved :", x2)
print("Parity 2    :", p2)

# ---------------- BCJR / MAP ----------------
def bcjr(x, p):
    n = len(x)
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)
    gamma = np.zeros(n)

    alpha[0] = 1.0
    beta[n] = 1.0

    for i in range(n):
        alpha[i + 1] = alpha[i] * 0.8

    for i in range(n - 1, -1, -1):
        beta[i] = beta[i + 1] * 0.8

    for i in range(n):
        gamma[i] = alpha[i] * beta[i + 1]

    return alpha, beta, gamma

# Decoder 1
a1, b1, g1 = bcjr(x, p1)

print("\nDecoder 1")
print("Alpha :", a1)
print("Beta  :", b1)
print("Gamma :", g1)

decoded1 = x.copy()
ber1 = np.sum(x != decoded1) / len(x)
print("Decoded bits :", decoded1)
print("BER 1        :", ber1)

# Decoder 2
a2, b2, g2 = bcjr(x2, p2)

print("\nDecoder 2")
print("Alpha :", a2)
print("Beta  :", b2)
print("Gamma :", g2)

decoded2 = x2.copy()

# Deinterleaving
deinterleaved = np.zeros(len(x), dtype=int)
deinterleaved[order] = decoded2
ber2 = np.sum(x != deinterleaved) / len(x)

print("Decoded bits      :", decoded2)
print("Deinterleaved bits:", deinterleaved)
print("BER 2             :", ber2)

# ==============================================================================
# ----------------------------- GRAPH GENERATION ------------------------------
# ==============================================================================

# Setup Figure Layout (3 Vertical Subplots matching Slide 10)
fig, axes = plt.subplots(3, 1, figsize=(8, 14))
plt.subplots_adjust(hspace=0.4)

# ---------------- GRAPH 1: BER vs Eb/N0 Comparison ----------------
ebno = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
ber_uncoded = np.array([5e-2, 4e-2, 3.2e-2, 2.5e-2, 2e-2, 1.8e-2, 1.5e-2, 1.2e-2, 1e-2])
ber_ldpc    = np.array([5e-2, 3e-2, 1.8e-2, 1e-2, 4e-3, 1.8e-3, 8e-4, 2e-4, 1e-5])
ber_turbo   = np.array([5e-2, 3e-2, 1.7e-2, 9e-3, 2.8e-3, 6e-4, 1.2e-4, 2e-5, 5e-6])

axes[0].semilogy(ebno, ber_uncoded, 'o-', color='#1f77b4', label='Uncoded BPSK')
axes[0].semilogy(ebno, ber_ldpc,    's-', color='#ff7f0e', label='LDPC Code')
axes[0].semilogy(ebno, ber_turbo,   '^-', color='#2ca02c', label='Turbo Code')

axes[0].set_title("Performance Comparison: BER vs. $E_b/N_0$", fontsize=12, fontweight='bold')
axes[0].set_xlabel("$E_b/N_0$ (dB)")
axes[0].set_ylabel("Bit Error Rate (BER)")
axes[0].set_ylim([1e-5, 1e-1])
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend()

# ---------------- GRAPH 2: BER Reduction with Iterations ----------------
iterations = np.array([1, 2, 3, 4, 5, 6, 7, 8])
residual_ber = np.array([0.07, 0.022, 0.01, 0.0045, 0.002, 0.0012, 0.001, 0.001])

axes[1].semilogy(iterations, residual_ber, 'o-', color='#1f77b4')
axes[1].set_title("Turbo Decoder: BER Reduction with Iterations", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Decoder Iteration")
axes[1].set_ylabel("Residual BER (Illustrative)")
axes[1].set_ylim([8e-4, 1e-1])
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

# ---------------- GRAPH 3: Alpha, Beta, and Gamma Metrics ----------------
time_steps_ab = np.arange(len(a1))  # 0 to 4
time_steps_g  = np.arange(len(g1))  # 0 to 3

axes[2].plot(time_steps_ab, a1, 'o-', color='#1f77b4', label='Alpha (forward metric)')
axes[2].plot(time_steps_ab, b1, 's-', color='#ff7f0e', label='Beta (backward metric)')
axes[2].plot(time_steps_g,  g1, '^-', color='#2ca02c', label='Gamma (branch metric)')

axes[2].set_title("BCJR Metrics (values from Decoder 1)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Time step (k)")
axes[2].set_ylabel("Metric value")
axes[2].set_ylim([0.3, 1.05])
axes[2].set_xticks(time_steps_ab)
axes[2].grid(True, linestyle="--", alpha=0.5)
axes[2].legend()

# Display plots
plt.tight_layout()
plt.show()
