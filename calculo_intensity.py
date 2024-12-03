import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from io import StringIO
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def shirley_background(x_data, y_data, init_back, end_back, n_iterations=6):
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
55901.0 1
56400.0 2
56553.0 3
57193.0 4
57592.0 5
57494.0 6
58685.0 7
58718.0 8
59453.0 9
59893.0 10
60524.0 11
61685.0 12
63105.0 13
64078.0 14
65765.0 15
68337.0 16
70873.0 17
74564.0 18
76521.0 19
76281.0 20
73342.0 21
67580.0 22
61213.0 23
55164.0 24
51596.0 25
49404.0 26
48372.0 27
47573.0 28
47159.0 29
47015.0 30
46589.0 31
46670.0 32
46316.0 33
46308.0 34
"""

# Conversão dos dados para DataFrame
df = pd.read_csv(StringIO(data), sep='\s+', header=None, names=['Y', 'X'])
def poly_fit(x, y, degree=3):
    coefficients = np.polyfit(x, y, degree)
    return coefficients
def smooth(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)
# Ordenar os dados por X em ordem crescente
df = df.sort_values(by='X', ascending=True)

# Separar X e Y
x = df['X'].values
y = df['Y'].values

y_smoothed = smooth(y, sigma=1)



# Definir os índices initback e endback
init_back = 1  # Ajuste conforme seu critério
end_back = len(x) - 1  # Ajuste conforme seu critério

# Aplicar o fundo Shirley
shirley_bg = shirley_background(x, y_smoothed, init_back, end_back)

# Ajustar o fundo para começar no valor de 30.000 (ou outro valor conforme necessário)
shirley_bg_adjusted = shirley_bg + (y[0] - shirley_bg[0])

# Calcula o fundo Shirley
bg = shirley_background(x, y_smoothed, init_back, end_back)

# Corrige os valores de intensidade
y_corrected = y_smoothed - bg
# Filtra os valores positivos de y_corrected
positive_values = np.where(y_corrected > 0, y_corrected, 0)
# Calcula a área apenas para os valores positivos
total_area = trapezoid(positive_values, x)
# Imprimir a área total
print(f'Área total corrigida: {total_area}')

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
