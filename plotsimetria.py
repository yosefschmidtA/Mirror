import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Função para processar os dados
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inicializar uma lista para armazenar os dados
    data = []
    theta_value = None  # Para armazenar o valor de θ

    # Iterar sobre as linhas, começando após o cabeçalho
    for i in range(17, len(lines)):
        line = lines[i].strip()  # Remove espaços em branco das extremidades

        if line:  # Verifica se a linha não está vazia
            parts = line.split()  # Divide a linha em partes

            if len(parts) == 6:  # Linha que contém θ
                theta_value = float(parts[3])  # Coleta θ na quarta posição

            elif len(parts) == 4 and theta_value is not None:  # Linha que contém φ e intensidade
                phi = float(parts[0])  # Coleta φ na primeira posição
                col1 = float(parts[1])  # Coleta o valor da segunda coluna
                col2 = float(parts[2])  # Coleta o valor da terceira coluna
                intensity = float(parts[3])  # Coleta intensidade na quarta posição
                data.append([phi, col1, col2, theta_value, intensity])

    # Criar um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])

    # Replicar os dados para cobrir os 360 graus
    df_0_90 =df.copy()
    df_0_90['Phi'] = 360 - df_0_90['Phi']

    df_90_180 = df.copy()
    df_90_180['Phi'] = 90 - df_90_180['Phi']  # Reflete os valores de Phi para o segundo quadrante

    df_180_270 = df.copy()
    df_180_270['Phi'] = 180 - df_180_270['Phi']  # Reflete os valores de Phi para o terceiro quadrante

    df_270_360 = df.copy()
    df_270_360['Phi'] = 270 - df_270_360['Phi']  # Reflete os valores de Phi para o quarto quadrante

    # Combinar os dados para cobrir os 4 quadrantes
    df_full = pd.concat([df,df_0_90, df_90_180, df_180_270, df_270_360]).reset_index(drop=True)

    return df_full

# Função para gerar o gráfico polar
def plot_polar(df):
    # Converter Phi de graus para radianos
    phi_rad = np.radians(df['Phi'])

    # Configurar o gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plotar os dados de intensidade
    sc = ax.scatter(phi_rad, df['Intensity'], c=df['Intensity'], cmap='inferno', s=20, marker='o')

    # Adicionar título e configurações
    ax.set_title('Gráfico Polar de Intensidade', va='bottom')
    ax.set_xlabel('Ângulo Phi (graus)')
    ax.set_ylabel('Intensidade')

    # Adicionar uma barra de cores
    plt.colorbar(sc, ax=ax, label='Intensidade')

    # Exibir o gráfico
    plt.show()

# Caminho do arquivo de dados
file_path = 'exp_Fe2P_fitted.out'  # Altere para o caminho real do seu arquivo

# Processar os dados
df = process_file(file_path)

# Plotar o gráfico polar
plot_polar(df)
