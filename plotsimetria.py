import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Função para processar o novo arquivo com blocos
def process_block_file(file_path):
    """
    Processa um arquivo organizado em blocos de Theta, com colunas Phi e Intensidade.
    """
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    theta_value = None  # Variável para armazenar o valor atual de Theta
    for line in lines:
        line = line.strip()
        if line.startswith("theta"):  # Identifica o início de um bloco
            # Extrai o valor de Theta
            theta_value = float(line.split()[1])
        elif line and theta_value is not None:  # Processa as linhas de dados no bloco
            # Cada linha deve ter pelo menos 4 colunas (Phi na coluna 0, Intensidade na coluna 3)
            parts = line.split()
            phi = float(parts[0])  # Phi na coluna 0
            intensity = float(parts[3])  # Intensidade na coluna 3
            data.append([phi, theta_value, intensity])

    # Criar um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Theta', 'Intensity'])
    return df

# Função para interpolar os dados
def interpolate_data(df, resolution=1000):
    """
    Interpola os dados para uma grade regular.
    """
    # Converter graus para radianos para o gráfico polar
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
def plot_polar_interpolated(df, resolution=1000):
    """
    Gera um gráfico polar interpolado com os dados fornecidos.
    """
    # Interpolar os dados
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)

    # Criando o gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plotando a intensidade interpolada
    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='gouraud', cmap='hot')

    # Ajuste para mostrar os valores reais de theta (em graus)
    ax.set_theta_offset(0)  # Ajusta o ponto de origem para o topo
    ax.set_theta_direction(1)  # Ajusta a direção dos ângulos para ser anti-horária

    # Limitar o eixo radial até o maior valor de theta
    max_theta = df['Theta'].max()  # Maior valor de theta presente nos dados
    ax.set_ylim(0, np.radians(max_theta))

    # Adicionar rótulos para os ângulos theta
    theta_ticks = np.linspace(0, max_theta, num=6)  # Até 6 ticks no eixo radial
    ax.set_yticks(np.radians(theta_ticks))
    ax.set_yticklabels([f'{int(tick)}°' for tick in theta_ticks], fontsize=12)

    # Adicionando a barra de cores
    fig.colorbar(c, ax=ax, label='Intensidade')

    plt.show()

# Uso da função
file_path = 'exp_Fe2_GaO_with_symmetry_0_180_blocks.txt'  # Substitua pelo caminho do novo arquivo
df = process_block_file(file_path)  # Processa o arquivo em blocos

# Gerar o gráfico polar interpolado
plot_polar_interpolated(df)
