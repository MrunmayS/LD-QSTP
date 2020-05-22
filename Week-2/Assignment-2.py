import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fivethirtyeight') 

x = list(range(50))
y = np.random.randn(50)

index_x = iter(x)
index_y = iter(y)


x_vals = []
y_vals = []

def animate(i):
    
    x_vals.append(next(index_x))
    y_vals.append(next(index_y))
    
    plt.cla() 
    plt.title('Live plot')

    if len(x_vals) < 10:
        plt.plot(x_vals, y_vals, marker='o', label='Live plot')
    else:
        plot_x = x_vals[len(x_vals)-10:]
        plot_y = y_vals[len(x_vals)-10:]
        plt.plot(plot_x, plot_y, marker='o', label='Live plot')
        
    plt.legend(loc='upper left')
    
    plt.tight_layout()\

ani = FuncAnimation(plt.gcf(), animate, interval=1000) 
plt.show()