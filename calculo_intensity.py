import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks


def shirley_background(x, y, initback, endback, max_iter=10000, tol=1e-6):
    """
    Calcula o fundo Shirley para os dados de XPS com base no procedimento fornecido.

    Parâmetros:
    - x: array-like, valores do eixo X (e.g., energia de ligação)
    - y: array-like, valores do eixo Y (e.g., intensidade)
    - initback: índice inicial para o cálculo do fundo
    - endback: índice final para o cálculo do fundo
    - max_iter: número máximo de iterações (padrão: 10000)
    - tol: tolerância para convergência (padrão: 1e-6)

    Retorna:
    - bg: array, o fundo calculado para cada ponto de x
    """
    bg = np.zeros_like(y)
    background0 = np.zeros_like(y)
    for iteration in range(max_iter):
        new_bg = np.copy(bg)

        # Cálculo do fundo Shirley
        a = y[initback]
        b = y[endback]
        for i in range(initback, endback, -1):
            sum1 = np.sum(y[initback:i] - background0[initback:i])
            sum2 = np.sum(y[initback:endback] - background0[initback:endback])
            new_bg[i] = (a - b) * (sum1 / sum2) + b

        # Ajuste para os pontos antes do initback e depois do endback
        new_bg[:initback] = new_bg[initback]
        new_bg[endback:] = new_bg[endback]

        # Verificando convergência
        if np.max(np.abs(new_bg - bg)) < tol:
            break

        bg = new_bg
        background0 = np.copy(bg)

    return bg


# Dados fornecidos pelo usuário
data = """
29769 1123.99988
30481 1123.49988
31333 1122.99988
32358 1122.49988
32797 1121.99988
35311 1121.49988
40265 1120.99988
48905 1120.49988
61725 1119.99988
79495 1119.49988
90546 1118.99988
87013 1118.49988
71519 1117.99988
55532 1117.49988
44807 1116.99988
39643 1116.49988
34716 1115.99988
31517 1115.49988
30025 1114.99988
29731 1114.49988
28387 1113.99988
"""

# Conversão dos dados para DataFrame
from io import StringIO
import pandas as pd

df = pd.read_csv(StringIO(data), sep='\s+', header=None, names=['Y', 'X'])

# Ordenar os dados por X em ordem crescente
df = df.sort_values(by='X', ascending=True)

# Separar X e Y
x = df['X'].values
y = df['Y'].values

# Definir os índices initback e endback
initback = 10  # Ajuste conforme seu critério
endback = len(x) - 10  # Ajuste conforme seu critério

# Aplicar o fundo Shirley
shirley_bg = shirley_background(x, y, initback, endback)

# Ajustar o fundo para começar no valor de 30.000 (ou outro valor conforme necessário)
shirley_bg_adjusted = shirley_bg + (y[0] - shirley_bg[0])

# Subtrair fundo Shirley ajustado do espectro
y_corrected = y - shirley_bg_adjusted

# Calcular a área total abaixo da curva corrigida (sem limitações)
total_area = trapezoid(np.abs(y_corrected), x)

# Imprimir a área total
print(f'Área total corrigida: {total_area}')

# Detecção de picos (se desejar destacar os picos)
peak_threshold = np.max(y_corrected)

# Encontrar os picos no espectro corrigido
picos, _ = find_peaks(y_corrected, height=peak_threshold)

# Plotagem do espectro original, fundo Shirley e espectro corrigido com picos
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original', marker='o')
plt.plot(x, shirley_bg_adjusted, label='Fundo Shirley Ajustado', linestyle='--')
plt.plot(x, y_corrected, label='Corrigido', marker='x')

# Preencher toda a área abaixo do espectro corrigido com amarelo
plt.fill_between(x, y_corrected, color='yellow', alpha=0.5)

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
