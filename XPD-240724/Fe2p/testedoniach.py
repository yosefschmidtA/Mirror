import numpy as np
import matplotlib.pyplot as plt

def doniach_sunjic(x, amp, mean, gamma, beta):
    # Garantir que o denominador não seja zero
    denom = (x - mean) ** 2 + gamma ** 2
    return (amp / np.pi) * (gamma / denom) * (1 + beta * (x - mean) / denom)

# Parâmetros para o ajuste
amp = 10000
mean = 10
gamma = 2
beta = 0.1

# Gerar valores de x para plotar
x_values = np.linspace(5, 15, 100)

# Gerar o pico Doniach-Sunjic
y_values = doniach_sunjic(x_values, amp, mean, gamma, beta)

# Plotar o pico Doniach-Sunjic
plt.plot(x_values, y_values, label='Doniach-Sunjic')
plt.xlabel('Energia de Ligação (eV)')
plt.ylabel('Intensidade')
plt.title('Pico Doniach-Sunjic')
plt.legend()
plt.grid(True)
plt.show()
