import numpy as np

import xy.pyplot as plt

x = np.linspace(-10.0, 10.0, 200)
ax = plt.gca()
ax.plot(x, 1e-9 + (x - x.min()) * 1e-10)
ax.set_yscale("logit")
ax.set_ylim(1e-10, 0.5)
ax.set_title("logit p-value decades")
