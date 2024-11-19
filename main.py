import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Função para processar o arquivo
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
                intensity = float(parts[3])  # Coleta intensidade na quarta posição
                data.append([phi, theta_value, intensity])

    # Cria um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Theta', 'Intensity'])
    return df


# Função para gerar o gráfico polar
def plot_polar(df):
    # Convertendo Phi para radianos
    phi = np.radians(df['Phi'])
    theta = np.radians(df['Theta'])  # Theta em radianos
    intensity = df['Intensity']

    # Criando um gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plotando os dados no gráfico polar
    sc = ax.scatter(phi, theta, c=intensity, cmap='viridis', s=10)  # Usando viridis para a coloração de intensidade
    ax.set_xlabel('Phi (radianos)')
    ax.set_ylabel('Theta (radianos)')

    # Adicionando a barra de cores
    fig.colorbar(sc, ax=ax, label='Intensidade')

    plt.show()


# Uso da função
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df = process_file(file_path)

# Gerar o gráfico polar
plot_polar(df)
