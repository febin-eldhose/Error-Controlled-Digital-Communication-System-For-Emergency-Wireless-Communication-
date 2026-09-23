import numpy as np
import matplotlib.pyplot as plt

N = 4000
snr_db = [-2, 0, 3, 5, 10]

bits = np.random.randint(0, 2, N)
bits = bits[:N-N%4]


def qpsk(data, snr_db):

    x = data.reshape(-1, 2)
    s = (1-2*x[:,0]) + 1j*(1-2*x[:,1])

    snr = 10**(snr_db/10)
    noise = np.sqrt(1/(2*snr)) * (
        np.random.randn(len(s)) +
        1j*np.random.randn(len(s))
    )

    r = s + noise

    return np.column_stack(
        ((r.real < 0), (r.imag < 0))
    ).astype(int).reshape(-1)


# QPSK
ber1 = []

for snr in snr_db:
    r = qpsk(bits, snr)
    ber1.append(np.mean(bits != r))


# Repetition
ber2 = []

for snr in snr_db:

    x = np.repeat(bits, 3)
    r = qpsk(x, snr)

    r = r.reshape(-1, 3)
    r = (np.sum(r, axis=1) >= 2).astype(int)

    ber2.append(np.mean(bits != r[:N]))


# Hamming (7,4)
ber3 = []

for snr in snr_db:

    data = bits.reshape(-1,4)
    coded = []

    for a,b,c,d in data:

        p1 = a^b^d
        p2 = a^c^d
        p4 = b^c^d

        coded += [p1,p2,a,p4,b,c,d]

    r = qpsk(np.array(coded), snr)
    r = r.reshape(-1,7)

    out = []

    for x in r:

        e = (x[0]^x[2]^x[4]^x[6]) \
            + 2*(x[1]^x[2]^x[5]^x[6]) \
            + 4*(x[3]^x[4]^x[5]^x[6])

        if e:
            x[e-1] ^= 1

        out += [x[2],x[4],x[5],x[6]]

    ber3.append(np.mean(bits != np.array(out)))


plt.semilogy(snr_db, ber1, 'o-', label='QPSK')
plt.semilogy(snr_db, ber2, 's-', label='QPSK + Repetition')
plt.semilogy(snr_db, ber3, '^-', label='QPSK + Hamming')

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('QPSK Reliability Comparison')
plt.grid()
plt.legend()
plt.show()