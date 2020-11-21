import numpy as np
import matplotlib.pyplot as plt

def f2(x,w):
    return (x-w)*x*(x+2)


x = np.linspace(-3,3,100)

plt.plot(x, f2(x,2), color='black', label='Sw=2s')
plt.plot(x, f2(x,1), color='cornflowerblue', label='Sw-1s')
plt.legend(loc='best')
plt.title('$f_2(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True)
