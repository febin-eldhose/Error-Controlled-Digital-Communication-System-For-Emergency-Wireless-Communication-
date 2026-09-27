import numpy as np
import matplotlib.pyplot as plt

# Number of information bits
N = 4000

# SNR values
snr_db = [-2, 0, 3, 5, 10]

# Generate random input bits
bits = np.random.randint(0, 2, N)


# --------------------------------------------------
# QPSK Modulation + AWGN Channel + Demodulation
# --------------------------------------------------

def qpsk(data, snr_db):

    # Group bits into pairs
    x = data.reshape(-1, 2)

    # QPSK mapping
    # 0 -> +1
    # 1 -> -1
    #
    # I = first bit
    # Q = second bit
    s = (1 - 2*x[:, 0]) + 1j*(1 - 2*x[:, 1])

    # Convert dB to linear scale
    snr = 10**(snr_db/10)

    # Generate complex Gaussian noise
    noise = np.sqrt(1/(2*snr)) * (
        np.random.randn(len(s)) +
        1j*np.random.randn(len(s))
    )

    # Received QPSK signal
    r = s + noise

    # QPSK demodulation
    received_bits = np.column_stack(
        (
            (r.real < 0),
            (r.imag < 0)
        )
    ).astype(int).reshape(-1)

    return received_bits


# --------------------------------------------------
# Convolutional Encoder
# --------------------------------------------------

def conv_encode(data):

    # Two memory elements
    state = 0

    coded = []

    for bit in data:

        # Generate two output bits
        b1 = bit ^ ((state >> 1) & 1) ^ (state & 1)
        b2 = bit ^ (state & 1)

        coded += [b1, b2]

        # Update encoder state
        state = ((state << 1) | bit) & 3

    return np.array(coded)


# --------------------------------------------------
# Viterbi Decoder
# --------------------------------------------------

def viterbi_decode(received):

    # 2 memory bits -> 2^2 = 4 states
    states = 4

    # Every input bit produces 2 coded bits
    n = len(received) // 2

    # Initially, state 0 is known
    metric = np.ones(states) * 100000
    metric[0] = 0

    # Store possible paths
    paths = [[] for _ in range(states)]

    # Process received bits pair by pair
    for i in range(n):

        r1 = received[2*i]
        r2 = received[2*i + 1]

        new_metric = np.ones(states) * 100000
        new_paths = [[] for _ in range(states)]

        # Check every possible current state
        for state in range(states):

            if metric[state] >= 100000:
                continue

            # Try both possible input bits
            for bit in [0, 1]:

                # Expected encoder output
                out1 = bit ^ ((state >> 1) & 1) ^ (state & 1)
                out2 = bit ^ (state & 1)

                # Calculate next state
                next_state = ((state << 1) | bit) & 3

                # Hamming distance
                error = (r1 != out1) + (r2 != out2)

                # New path metric
                new_value = metric[state] + error

                # Keep the path with minimum error
                if new_value < new_metric[next_state]:

                    new_metric[next_state] = new_value

                    new_paths[next_state] = (
                        paths[state] + [bit]
                    )

        metric = new_metric
        paths = new_paths

    # Select the path having minimum total error
    best_state = np.argmin(metric)

    return np.array(paths[best_state])


# --------------------------------------------------
# QPSK + Convolutional Coding
# --------------------------------------------------

ber_conv = []

for snr in snr_db:

    # 1. Convolutional encoding
    coded = conv_encode(bits)

    # 2. QPSK transmission through noisy channel
    received = qpsk(coded, snr)

    # 3. Viterbi decoding
    decoded = viterbi_decode(received)

    # Make sure length matches original data
    decoded = decoded[:N]

    # 4. Calculate BER
    ber = np.mean(bits != decoded)

    ber_conv.append(ber)


# --------------------------------------------------
# Plot BER
# --------------------------------------------------

plt.semilogy(
    snr_db,
    ber_conv,
    'd-',
    label='QPSK + Convolutional Coding'
)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')

plt.title('QPSK + Convolutional Coding')

plt.grid(True)
plt.legend()

plt.show()
