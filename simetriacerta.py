import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# Função para processar os dados
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    theta_value = None

    for i in range(17, len(lines)):
        line = lines[i].strip()

        if line:
            parts = line.split()

            if len(parts) == 6:
                theta_value = float(parts[3])

            elif len(parts) == 4 and theta_value is not None:
                phi = float(parts[0])
                col1 = float(parts[1])
                col2 = float(parts[2])
                intensity = float(parts[3])
                data.append([phi, col1, col2, theta_value, intensity])

    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])

    # Replicar os dados para cobrir os 360 graus
    df_0_90 = df.copy()
    df_0_90['Phi'] = 90 + df_0_90['Phi']

    df_90_180 = df.copy()
    df_90_180['Phi'] = 180 + df_90_180['Phi']

    df_180_270 = df.copy()
    df_180_270['Phi'] = 270 + df_180_270['Phi']

    df_full = pd.concat([df, df_0_90, df_90_180, df_180_270]).reset_index(drop=True)

    return df_full


# Função para interpolar os dados
def interpolate_data(df, resolution=1000):
    # Converter Phi para radianos
    phi_vals = np.radians(df['Phi'])
    theta_vals = np.radians(df['Theta'])

    # Usar a faixa de Phi original sem limitação fixa
    phi_min, phi_max = np.min(phi_vals), np.max(phi_vals)

    # Criar um grid para interpolação, abrangendo todo o intervalo de Phi
    phi_grid, theta_grid = np.meshgrid(np.linspace(phi_min, phi_max, resolution), np.linspace(0, np.pi / 2, resolution))

    # Interpolando a intensidade usando griddata
    intensity_grid = griddata((phi_vals, theta_vals), df['Intensity'], (phi_grid, theta_grid), method='linear')

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

    ax.set_xlabel('Phi (graus)')
    ax.set_ylabel('Theta (graus)')

    # Adicionando a barra de cores
    fig.colorbar(c, ax=ax, label='Intensidade')

    # Exibir o gráfico
    plt.show()


# Caminho do arquivo de dados
file_path = 'exp_Fe2P_fitted.out'

# Processar os dados
df = process_file(file_path)

# Plotar o gráfico polar interpolado
plot_polar_interpolated(df)
