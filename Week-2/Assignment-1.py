import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

domain = np.linspace(-6,6,1000)
plt.title("Normal Distribution")
plt.plot(domain, norm.pdf(domain,0,np.sqrt([5])), label = "Mean = 0, Var = 5")
plt.plot(domain, norm.pdf(domain,4,np.sqrt([.2])), label = "Mean = 4, Var = 0.2")
plt.plot(domain, norm.pdf(domain, -3.5, pow(0.05,0.5)), label = "Mean = -3.5, Var = 0.05")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()