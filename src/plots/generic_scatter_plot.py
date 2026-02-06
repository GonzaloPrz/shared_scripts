import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os
# Generar datos similares
np.random.seed(2)
x = np.random.uniform(0, 10, size=30)
y = 0.9 * x + np.random.normal(0,2, size=30)

# Ajustar regresión
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred_line = model.predict(np.linspace(0, 10, 100).reshape(-1, 1))

# Plot
plt.figure(figsize=(3.5, 3))
plt.scatter(x, y, color='#c9a400')  # amarillo oscuro
plt.plot(np.linspace(0, 10, 100), y_pred_line, color='black')

plt.xlabel('Predicted value', fontsize=16, weight='bold')
plt.ylabel('Real value', fontsize=16, weight='bold')
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig(Path(Path.home(),'generic_scatter_plot2.png'), dpi=300, bbox_inches='tight')
