
'''import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4])
l= 4
n = np.arange(l)
omega = np.linspace(-np.pi, np.pi)  
X = np.array([sum(x[n]* np.exp(-1j * w * n)
for n in range(l)) for w in omega])
plt.subplot(2, 1, 1)
plt.plot(omega, np.abs(X))
plt.title('Magnitude of DTFT')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.subplot(2, 1, 2)
plt.plot(omega, np.angle(X))
plt.title('Phase of DTFT')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
def dtft(x):
	n=len(x)
	w=np.linspace(-np.pi,np.pi,n) #frequency vector to create n evenly spaced  points b/w -pi to pi
	X=np.zeros(n) #initialize dtft array X
	for k in range(n):
		X[k]=np.sum(x*np.exp(-1j*k*w))
	return X,w
x=np.array([1,2,3,4,5])
X,w=dtft(x)
plt.subplot(121)
plt.plot(w,np.abs(X))
plt.xlabel("Frequency")
plt.ylabel('magnitude')
plt.title('magnitude spectrum')

plt.subplot(122)
plt.plot(w,np.angle(X))
plt.xlabel("Frequency")
plt.ylabel('phase')
plt.title('phase spectrum')
plt.show()
