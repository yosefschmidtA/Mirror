import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import false_discovery_control


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
                data.append([phi, col1, col2, theta_value, intensity, True])  # Marcar como original

    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity', 'IsOriginal'])
    # Verificar o intervalo de Phi
    phi_min = df['Phi'].min()
    phi_max = df['Phi'].max()
    phi_interval = phi_max - phi_min

    if phi_interval < 360 and df['Phi'].max() < 360:
        # Encontrar os pontos com Phi = 0 e duplicar como Phi = 360
        df_360 = df[df['Phi'] == 180].copy()
        df_360['Phi'] = 360
        df = pd.concat([df, df_360], ignore_index=True)

    if phi_interval == 120:
        df_0_120 = df.copy()
        df_0_120['Phi'] = 120 + df_0_120['Phi']
        df_0_120['isOriginal'] = False

        df_240_360 = df.copy()
        df_240_360['Phi'] = 240 + df_240_360['Phi']

        df_full = pd.concat([df, df_0_120, df_240_360]).reset_index(drop=True)
        return df_full

    if phi_interval == 90:
        # Replicar os dados para cobrir os 360 graus (marcando-os como não originais)
        df_0_90 = df.copy()
        df_0_90['Phi'] = 90 + df_0_90['Phi']
        df_0_90['IsOriginal'] = False

        df_90_180 = df.copy()
        df_90_180['Phi'] = 180 + df_90_180['Phi']
        df_90_180['IsOriginal'] = False

        df_180_270 = df.copy()
        df_180_270['Phi'] = 270 + df_180_270['Phi']
        df_180_270['IsOriginal'] = False
        df_full = pd.concat([df, df_0_90, df_90_180, df_180_270]).reset_index(drop=True)
        return df_full

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
def rotate_phi(df, rotation_angle):
    """
    Rotaciona os valores de Phi no DataFrame.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com a coluna 'Phi'.
    - rotation_angle (float): Ângulo de rotação em graus.

    Retorna:
    - pd.DataFrame: DataFrame com Phi rotacionado.
    """
    df['Phi'] += rotation_angle
    return df


# Caminho do arquivo de dados
file_path = 'coeficientes_ajustados.txt'


def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, onde cada bloco corresponde a um valor de θ (Theta).
    """
    # Filtrar apenas os dados originais
    df_original = df[df['IsOriginal']]

    num_theta = df_original['Theta'].nunique()  # Número de ângulos theta únicos
    num_phi = df_original['Phi'].nunique()  # Número de ângulos phi únicos
    num_points = len(df_original)  # Total de pontos

    with open(file_name, 'w') as file:
        # Cabeçalho inicial que aparece uma vez no arquivo
        file.write(f"      {num_theta}    {num_points}    0     datakind beginning-row linenumbers\n")
        file.write(f"MSCD Version 1.00 Yufeng Chen and Michel A Van Hove\n")
        file.write(f"Lawrence Berkeley National Laboratory (LBNL), Berkeley, CA 94720\n")
        file.write(f"Copyright (c) Van Hove Group 1997. All rights reserved\n")
        file.write(f"--------------------------------------------------------------\n")
        file.write(f"angle-resolved photoemission extended fine structure (ARPEFS)\n")
        file.write(f"experimental data for Fe 2p3/2 from Fe on STO(100)  excited with hv=1810eV\n")
        file.write(f"provided by Pancotti et al. (LNLS in 9, June 2010)\n")
        file.write(f"   initial angular momentum (l) = 1\n")
        file.write(f"   photon polarization angle (polar, azimuth) = (  30.0,   0.0 ) (deg)\n")
        file.write(f"   sample temperature = 300 K\n")
        file.write(f"   photoemission angular scan curves\n")
        file.write(f"     (curve point theta phi weightc weighte//k intensity chiexp)\n")  # Cabeçalho dinâmico
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")

        # Número do bloco de θ
        number_of_theta = 0

        # Loop para os diferentes valores de θ
        for theta in sorted(df_original['Theta'].unique()):
            # Cabeçalho dinâmico do bloco de θ
            number_of_theta += 1
            first_row = df_original[df_original['Theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {theta:.4f}      1.00000      0.00000\n")

            # Dados para o θ atual
            subset = df_original[df_original['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(
                    f"      {row['Phi']:.4f}      {row['Col1']:.2f}      {row['Col2']:.2f}      {row['Intensity']:.7f}\n")

            # Linha em branco para separar os blocos
            file.write("")


# Processar os dados
df = process_file(file_path)

df = rotate_phi(df, 0)


save_to_txt_with_blocks(df, 'dados_rotacionados.txt')

plot_polar_interpolated(df)