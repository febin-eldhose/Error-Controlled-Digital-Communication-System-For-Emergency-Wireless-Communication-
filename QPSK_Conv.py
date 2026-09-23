import numpy as np
import matplotlib.pyplot as plt

N = 10000
snr_db = [-2, 0, 3, 5, 10]

bits = np.random.randint(0,2,N)

def encode(bits):

    state = 0
    coded = []

    for b in bits:

        x1 = b ^ ((state>>1)&1) ^ (state&1)
        x2 = b ^ (state&1)

        coded += [x1,x2]
        state = ((state<<1)|b)&3

    return np.array(coded)

def decode(r):

    paths = {0:(0,[])}

    for i in range(0,len(r),2):

        new = {}

        for state,(metric,path) in paths.items():

            for b in [0,1]:

                x1 = b ^ ((state>>1)&1) ^ (state&1)
                x2 = b ^ (state&1)

                d = (r[i]!=x1)+(r[i+1]!=x2)

                ns = ((state<<1)|b)&3

                if ns not in new or metric+d < new[ns][0]:
                    new[ns] = (metric+d,path+[b])

        paths = new

    return np.array(min(paths.values(),key=lambda x:x[0])[1])

BER = []

for db in snr_db:

    coded = encode(bits)

    x = coded.reshape(-1,2)

    s = (1-2*x[:,0]) + 1j*(1-2*x[:,1])

    snr = 10**(db/10)

    noise = np.sqrt(1/(2*snr))*(
        np.random.randn(len(s))+
        1j*np.random.randn(len(s))
    )

    r = s + noise

    d1 = (r.real < 0).astype(int)
    d2 = (r.imag < 0).astype(int)

    received = np.column_stack((d1,d2)).reshape(-1)

    decoded = decode(received)

    BER.append(np.mean(bits != decoded))

plt.semilogy(snr_db,BER,'o-')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.title('QPSK with Convolutional Coding')
plt.grid()
plt.show()