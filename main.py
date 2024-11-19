import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


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


# Função para interpolar os dados
def interpolate_data(df, resolution=100):
    # Definir uma grade regular para a interpolação
    phi = np.radians(df['Phi'])
    theta = np.radians(df['Theta'])
    intensity = df['Intensity']

    # Criando uma grade de pontos onde queremos interpolar
    phi_grid = np.linspace(np.min(phi), np.max(phi), resolution)
    theta_grid = np.linspace(np.min(theta), np.max(theta), resolution)

    # Criando uma grade de malha para interpolação
    phi_grid, theta_grid = np.meshgrid(phi_grid, theta_grid)

    # Realizar a interpolação
    intensity_grid = griddata((phi, theta), intensity, (phi_grid, theta_grid), method='cubic')

    return phi_grid, theta_grid, intensity_grid


# Função para gerar o gráfico polar
def plot_polar_interpolated(df, resolution=100):
    # Interpolar os dados
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)

    # Criando o gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plotando a intensidade interpolada
    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='auto', cmap='viridis')

    ax.set_xlabel('Phi (radianos)')
    ax.set_ylabel('Theta (radianos)')

    # Adicionando a barra de cores
    fig.colorbar(c, ax=ax, label='Intensidade')

    plt.show()


# Uso da função
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df = process_file(file_path)

# Gerar o gráfico polar interpolado
plot_polar_interpolated(df)
