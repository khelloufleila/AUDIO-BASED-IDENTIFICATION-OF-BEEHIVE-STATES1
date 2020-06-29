#
#  Data augmentation functions
#
#  Author: dominique@fourer.fr
#  Date: 27th of may 2020
#
import numpy as np
import scipy as sc
from scipy.linalg import norm

## calcul du Signal-to-Noise ratio
def SNR(x, x_hat):
	return 20 * np.log10(norm(x) / norm(x-x_hat))

# fonction permettant de bruiter un signal pour obtenir le SNR voulu
def sigmerge(x, b, snr_target):
	Ex1=sc.mean(sc.power(abs(x),2));
	Ex2=sc.mean(sc.power(abs(b),2));
	h=sc.sqrt(Ex1/(Ex2*sc.power(10,snr_target/10.)));
	sig=x+h*b;
	return sig
	
	
# Test function


def data_augmentation(x):
	#x = sc.hanning(100);   ## the original signal
	b = sc.randn(1,22050);   ## noise signal
	snr_target = 20;       ## the desired SNR
	
	print("target SNR: "+str(snr_target))

	y = sigmerge(x, b, snr_target)      # merge the signals together
	print("estimated SNR=", SNR(x,y))   # estimate again snr
	return y

def test():
	x = sc.hanning(100);   ## the original signal
	b = sc.randn(1,100);   ## noise signal
	snr_target = 20;       ## the desired SNR
	
	print("target SNR: "+str(snr_target))

	y = sigmerge(x, b, snr_target)      # merge the signals together
	print("estimated SNR=", SNR(x,y))   # estimate again snr

