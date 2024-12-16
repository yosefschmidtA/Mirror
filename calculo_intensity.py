import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from io import StringIO
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def shirley_background(x_data, y_data, init_back, end_back, n_iterations=100):
    """
    Calcula o fundo de Shirley para um espectro de intensidade.

    Parameters:
    x_data (array): O vetor de energias de ligação (ou qualquer outra variável x).
    y_data (array): O vetor de intensidades do espectro.
    init_back (int): Índice inicial para calcular o fundo.
    end_back (int): Índice final para calcular o fundo.
    n_iterations (int): Número de iterações para refinar o fundo.

    Returns:
    background (array): O vetor de fundo calculado.
    """
    # Inicializa o vetor de fundo com os valores nas extremidades
    background = np.zeros_like(y_data)
    background0 = np.zeros_like(y_data)

    # Definindo os valores de intensidade nas extremidades
    a = y_data[init_back]
    b = y_data[end_back]

    # Calcula o fundo de Shirley por n iterações
    for nint in range(n_iterations):
        for k2 in range(end_back, init_back - 1, -1):
            sum1 = 0
            sum2 = 0
            for k in range(end_back, k2 - 1, -1):
                sum1 += y_data[k] - background0[k]
            for k in range(end_back, init_back - 1, -1):
                sum2 += y_data[k] - background0[k]

            # Calcula o fundo interpolado entre as extremidades
            background[k2] = (a - b) * sum1 / sum2 + b

        # Ajuste o fundo para as extremidades
        background[:init_back] = background[init_back]
        background[end_back:] = background[end_back]

        # Atualiza o fundo de referência para a próxima iteração
        background0 = background.copy()

    return background


# Dados fornecidos pelo usuário
data = """
29846.0 1
29959.0 2
30482.0 3
30104.0 4
31490.0 5
31852.0 6
32896.0 7
34780.0 8
35965.0 9
34554.0 10
31609.0 11
29685.0 12
27405.0 13
26666.0 14
26401.0 15
25505.0 16
25315.0 17
26364.0 18
"""

df = pd.read_csv(StringIO(data), sep='\s+', header=None, names=['Y', 'X'])

def smooth(data, sigma):
    return gaussian_filter1d(data, sigma=sigma)
# Ordenar os dados por X em ordem crescente
df = df.sort_values(by='X', ascending=True)

# Separar X e Y
x = df['X'].values
y = df['Y'].values

y_smoothed = smooth(y, sigma=1)



# Definir os índices initback e endback
init_back = 0  # Ajuste conforme seu critério
end_back = len(x) - 1  # Ajuste conforme seu critério

shirley_bg = shirley_background(x, y, init_back, end_back)

# Ajustar o fundo para começar no valor de 30.000 (ou outro valor conforme necessário)
shirley_bg_adjusted = shirley_bg + (y[0] - shirley_bg[0])

# Calcula o fundo Shirley
bg = shirley_background(x, y_smoothed, init_back, end_back)

# Corrige os valores de intensidade
y_corrected = y_smoothed - bg
positive_values = y_corrected.copy()
positive_values[positive_values < 0] = 0
# Calcula a área apenas para os valores positivos
total_area = trapezoid(positive_values, x)
# Imprimir a área total
print(f'Área total: {total_area}')

# Detecção de picos (se desejar destacar os picos)
peak_threshold = np.max(y_corrected)

# Encontrar os picos no espectro corrigido
picos, _ = find_peaks(y_corrected, height=peak_threshold)

# Plotagem do espectro original, fundo Shirley e espectro corrigido com picos
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original', marker='o')
plt.plot(x, shirley_bg_adjusted, label='Fundo Shirley', linestyle='--')
plt.plot(x, y_corrected, label='Corrigido', marker='x')

# Preencher toda a área abaixo do espectro corrigido com amarelo
plt.fill_between(x, positive_values, color='yellow', alpha=0.5)

plt.xlabel('Energia de Ligação (eV)')
plt.ylabel('Intensidade')
plt.title('Espectro XPS com Fundo Shirley Ajustado e Picos Detectados')
plt.legend()
plt.grid(True)
plt.show()

# Exibindo os picos detectados
print("Picos detectados (índices):", picos)
print("Valores de intensidade nos picos:", y_corrected[picos])
print("Valores de energia correspondentes aos picos:", x[picos])
