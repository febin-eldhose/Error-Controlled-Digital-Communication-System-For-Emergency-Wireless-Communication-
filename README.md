# Error-Controlled-Digital-Communication-System-For-Emergency-Wireless-Communication-

This project is an error controlled communication system that can be used in emergency situations. The main focus is finding a suitable combination of QPSK and error-control coding that gives low BER without excessive transmission overhead. 

 We have established the basic QPSK transmission system and are testing it over an AWGN channel by observing the BER for different signal-to-noise ratios. We are now working on adding error-control techniques so that the system can reliably transmit emergency information even when the channel contains noise.
We first experimented with repetition coding, where the transmitted information is repeated several times and majority voting is used at the receiver. Although this improves reliability, it requires additional bandwidth. We are therefore moving toward convolutional coding, where redundancy is added in a more structured way. The encoded data is QPSK modulated, transmitted through the noisy channel, demodulated at the receiver, and then decoded using the Viterbi algorithm.
The current stage is still a software simulation, and the main focus is finding a suitable combination of QPSK and error-control coding that gives low BER without excessive transmission overhead.

***We are still in a developing stage***
