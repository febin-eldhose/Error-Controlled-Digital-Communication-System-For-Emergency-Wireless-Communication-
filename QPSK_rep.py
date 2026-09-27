import numpy as np
import matplotlib.pyplot as plt

N = 10000
m = 3
snr_db = [-2, 0, 3, 5, 10]

bits = np.random.randint(0, 2, N)
bits = bits[:N-N%2]

BER = []

for db in snr_db:

    b = np.repeat(bits, m)
    b = b[:len(b)-len(b)%2]

    x = b.reshape(-1,2)

    s = (1-2*x[:,0]) + 1j*(1-2*x[:,1])

    snr = 10**(db/10)
    noise = np.sqrt(1/(2*snr)) * (
        np.random.randn(len(s)) +
        1j*np.random.randn(len(s))
    )

    r = s + noise

    d1 = (r.real < 0).astype(int)
    d2 = (r.imag < 0).astype(int)

    detected = np.column_stack((d1,d2)).reshape(-1)

    detected = detected.reshape(-1,m)

    final = (np.sum(detected,axis=1) >= (m+1)//2).astype(int)

    BER.append(np.mean(bits != final[:N]))

plt.semilogy(snr_db, BER, 'o-')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('QPSK with Repetition Coding')
plt.grid()
plt.show()
