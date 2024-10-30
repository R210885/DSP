'''import numpy as np
import matplotlib.pyplot as plt
def dtft(x):
    X=[]
    w=np.arange(-np.pi,np.pi,0.00001*np.pi)
    for f in w:
        s=0
        for n in range(len(x)):
            s+=x[n]*np.exp(-1j*f*n)
        X.append(s)
    return np.array(X)'''
import numpy as np
import matplotlib.pyplot as plt
def dtft(x):
	n=len(x)
	w=np.linspace(-np.pi,np.pi,n) #frequency vector to create n evenly spaced  points b/w -pi to pi
	X=np.zeros(n) #initialize dtft array X
	for k in range(n):
		X[k]=np.sum(x*np.exp(-1j*k*w) )
	return X,w

x1=np.array([2,3,4,5])
x2=np.array([3,6,7,8])
X1,w=dtft(x1)
X2,w=dtft(x2)
X3,w=dtft(x1+x2)
X4=X1+X2
if any(X4==X3):
	print("linearity is proved")
else:
	print("linearity is not proved")
print(X1)
print(X2)
print(X3)
print(X4)

