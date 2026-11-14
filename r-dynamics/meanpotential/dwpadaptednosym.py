import numpy as np
import matplotlib.pyplot as plt


L = 1.0
a = 1.0
delta_G = 0.1


c=np.linspace(-0.5, 1.5, 1000)


G=a * c**2 * (c-L)**2 + delta_G * c / L


plt.plot(c, G)


plt.xlabel("Molecular coordinate c")
plt.ylabel("Free energy G(c)")
plt.title("Non-Symmetric free energy landscape")

plt.show()
