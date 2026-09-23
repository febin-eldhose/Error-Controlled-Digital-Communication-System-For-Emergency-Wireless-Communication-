import numpy as np
import matplotlib.pyplot as plt

N = 10000
snr_db = [-2, 0, 3, 5, 10]

bits = np.random.randint(0, 2, N)
bits = bits[:N-N%4]

BER = []

for db in snr_db:

    data = bits.reshape(-1,4)
    coded = []

    for d in data:
        d1,d2,d3,d4 = d

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4

        coded.append([p1,p2,d1,p4,d2,d3,d4])

    coded = np.array(coded).reshape(-1)

    x = coded.reshape(-1,2)

    s = (1-2*x[:,0]) + 1j*(1-2*x[:,1])

    snr = 10**(db/10)

    noise = np.sqrt(1/(2*snr)) * (
        np.random.randn(len(s)) +
        1j*np.random.randn(len(s))
    )

    r = s + noise

    d1 = (r.real < 0).astype(int)
    d2 = (r.imag < 0).astype(int)

    received = np.column_stack((d1,d2)).reshape(-1)
    received = received.reshape(-1,7)

    corrected = []

    for x in received:

        p1 = x[0]^x[2]^x[4]^x[6]
        p2 = x[1]^x[2]^x[5]^x[6]
        p4 = x[3]^x[4]^x[5]^x[6]

        error = p1 + 2*p2 + 4*p4

        if error:
            x[error-1] ^= 1

        corrected.append(x)

    corrected = np.array(corrected)

    decoded = corrected[:,[2,4,5,6]].reshape(-1)

    BER.append(np.mean(bits != decoded))

plt.semilogy(snr_db, BER, 'o-')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('QPSK with Hamming (7,4) Coding')
plt.grid()
plt.show()