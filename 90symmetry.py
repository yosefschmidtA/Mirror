import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# Função para processar o arquivo
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inicializar uma lista para armazenar os dados
    data = []
    theta_value = None  # Para armazenar o valor de θ

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

    # Cria um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])
    return df


# Função para aplicar a simetria vertical nos dados
def apply_vertical_symmetry(df):
    df_original = df[df['Phi'] <= 90].copy()

    # Simetria para os outros quadrantes
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['Phi'] = 180 - df_original['Phi']

    df_symmetry_2 = df_symmetry_1.copy()
    df_symmetry_2['Phi'] = 360 - df_symmetry_1['Phi']

    df_symmetry_3 = df_original.copy()
    df_symmetry_3['Phi'] = 180 + df_original['Phi']

    # Combinar todos os dados
    df_combined = pd.concat([df_original, df_symmetry_1, df_symmetry_2, df_symmetry_3], ignore_index=True)

    # Ajustar os valores de Phi no intervalo [0, 360]
    df_combined['Phi'] = df_combined['Phi'] % 360
    df_combined = df_combined.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)

    return df_combined


# Função para interpolar os dados
def interpolate_data(df, resolution=500):
    phi = np.radians(df['Phi'])
    theta = np.radians(df['Theta'])
    intensity = df['Intensity']

    phi_grid = np.linspace(np.min(phi), np.max(phi), resolution)
    theta_grid = np.linspace(np.min(theta), np.max(theta), resolution)
    phi_grid, theta_grid = np.meshgrid(phi_grid, theta_grid)

    intensity_grid = griddata((phi, theta), intensity, (phi_grid, theta_grid), method='cubic')

    return phi_grid, theta_grid, intensity_grid


# Função para gerar o gráfico polar
def plot_polar(df, resolution=500):
    phi_grid, theta_grid, intensity_grid = interpolate_data(df, resolution)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))

    c = ax.pcolormesh(phi_grid, theta_grid, intensity_grid, shading='gouraud', cmap='viridis')
    fig.colorbar(c, ax=ax, label='Intensity')

    ax.set_theta_offset(np.pi / 2)  # Ajustar para começar no topo
    ax.set_theta_direction(-1)  # Direção dos ângulos

    ax.set_title("Polar Plot com Simetria", va='bottom')
    plt.show()


# Fluxo Principal
file_path = 'exp_Fe2P_fitted.out'  # Substitua pelo caminho do seu arquivo
df_original = process_file(file_path)

# Gerar a simetria para plotagem
df_with_symmetry = apply_vertical_symmetry(df_original)

# Plotar os dados com simetria
plot_polar(df_with_symmetry)
