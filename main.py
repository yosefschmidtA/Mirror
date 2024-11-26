import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


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

    # Cria um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])
    return df


def normalize_and_complete_phi(df):
    """
    Normaliza o intervalo de Phi para começar em 0 e completa os quadrantes simetricamente.
    """
    phi_min = df['Phi'].min()
    phi_max = df['Phi'].max()
    phi_range = phi_max - phi_min
    # Reescalar Phi para começar em 0 se necessário
    df['Phi'] = df['Phi'] - phi_min

    if phi_range < 360 and df['Phi'].max() < 360:
        # Encontrar os pontos com Phi = 0 e duplicar como Phi = 360
        df_360 = df[df['Phi'] == 180].copy()
        df_360['Phi'] = 360
        df = pd.concat([df, df_360], ignore_index=True)



        # Caso o intervalo de Phi seja menor que 360, completaremos os quadrantes simetricamente
        if phi_range <= 90:
            df_90_180 = df.copy()
            df_90_180['Phi'] = 180 - df_90_180['Phi']

            df_180_270 = df.copy()
            df_180_270['Phi'] = 180 + df_180_270['Phi']

            df_270_360 = df.copy()
            df_270_360['Phi'] = 360 - df_270_360['Phi']

            df_sobre = df[df['Phi'] == 180].copy()
            df_sobre['Phi'] = 360

            df = pd.concat([df, df_90_180, df_180_270, df_270_360, df_sobre], ignore_index=True)

        elif phi_range <= 180:
            df_180_360 = df.copy()
            df_180_360['Phi'] = 360 - df_180_360['Phi']
            df = pd.concat([df, df_180_360], ignore_index=True)

    # Ordenar para consistência
    df = df.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)
    return df


# Função para interpolar os dados
def interpolate_data(df, resolution=1000):
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
def plot_polar_interpolated(df, resolution=1000):
    # Interpolar os dados
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)

    # Criando o gráfico polar
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Plotando a intensidade interpolada
    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='gouraud', cmap='hot')

    # Ajuste para mostrar os valores reais de theta (em graus)
    ax.set_theta_offset(0)  # Ajusta o ponto de origem para o topo, mantendo o valor de 0° no topo
    ax.set_theta_direction(1)  # Ajusta a direção dos ângulos para ser anti-horária

    # Definir o limite máximo do eixo theta com base no maior valor de theta nos dados
    max_theta = df['Theta'].max()  # Maior valor de theta presente nos dados
    ax.set_ylim(0, np.radians(max_theta))  # Limitar o eixo radial até o maior valor de theta

    # Adiciona rótulos para os ângulos theta, ajustados conforme o máximo de theta nos dados
    theta_ticks = np.linspace(0, max_theta, num=6)  # Definir até 6 ticks no eixo theta
    ax.set_yticks(np.radians(theta_ticks))  # Converte para radianos
    ax.set_yticklabels([f'{int(tick)}°' for tick in theta_ticks], fontsize=12)  # Exibe como graus

    ax.set_xlabel('Phi ')
    ax.set_ylabel('... ')

    # Adicionando a barra de cores
    fig.colorbar(c, ax=ax, label='Intensidade')

    plt.show()


# Uso da função
file_path = 'exp_Fe2p_090610_theta35_fitted_V4.out'  # Substitua pelo caminho do seu arquivo
df = process_file(file_path)

df = normalize_and_complete_phi(df)

# Gerar o gráfico polar interpolado
plot_polar_interpolated(df)