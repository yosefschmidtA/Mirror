import pandas as pd


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

    num_theta = df['Theta'].nunique()
    num_points = len(data)
    initial_theta = df['Theta'].min()  # Ou outro critério baseado nos dados

    return df, num_theta, num_points, initial_theta


# Função para aplicar a simetria vertical nos dados
def apply_vertical_symmetry(df):
    # Filtrar dados de Phi entre 0 e 90
    df_original = df[df['Phi'] <= 90].copy()

    # Criar os dados simétricos para 90 a 180 (reflexão do primeiro quadrante)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['Phi'] = 180 - df_original['Phi']  # Espelhar Phi verticalmente para o segundo quadrante

    # Criar os dados simétricos para 180 a 270 (reflexão do segundo quadrante)
    df_symmetry_2 = df_symmetry_1.copy()
    df_symmetry_2['Phi'] = 360 - df_symmetry_1['Phi']  # Espelhar Phi verticalmente para o terceiro quadrante

    # Criar os dados simétricos para 270 a 360 (reflexão do primeiro quadrante)
    df_symmetry_3 = df_symmetry_1.copy()
    df_symmetry_3['Phi'] = 180 + df_symmetry_1['Phi']  # Espelhar Phi para o quarto quadrante

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1, df_symmetry_2, df_symmetry_3], ignore_index=True)

    # Garantir que os valores de Phi estejam dentro do intervalo [0, 360]
    df_combined = df_combined[(df_combined['Phi'] >= 0) & (df_combined['Phi'] <= 360)]

    # Marcar como extras os pontos adicionais (360 equivale ao 0)
    df_combined['IsExtra'] = df_combined['Phi'] == 360

    # Remover duplicatas para evitar contagens incorretas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)

    return df_combined


# Função para salvar o DataFrame em formato de bloco
def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, onde cada bloco corresponde a um valor de θ (Theta).
    """
    # Filtrar apenas os pontos que não são extras para cálculos
    valid_df = df[~df['IsExtra']]  # Considerar apenas os valores válidos
    num_theta = valid_df['Theta'].nunique()  # Número de ângulos theta únicos
    num_phi = valid_df['Phi'].nunique()  # Número de ângulos phi únicos
    num_points = num_theta * num_phi  # Total de pontos
    theta_initial = df['Theta'].min()  # Valor inicial de Theta

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
        file.write(f"   intial angular momentum (l) = 1\n")
        file.write(f"   photon polarization angle (polar,azimuth) = (  30.0,   0.0 ) (deg)\n")
        file.write(f"   sample temperature = 300 K\n")
        file.write(f"   photoemission angular scan curves\n")
        file.write(f"     (curve point theta phi weightc weighte//k intensity chiexp)\n")  # Cabeçalho dinâmico
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")
        number_of_theta = 0
        # Loop para os diferentes valores de θ
        for theta in df['Theta'].unique():
            # Cabeçalho dinâmico do bloco de θ
            number_of_theta += 1
            first_row = df[df['Theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {first_row['Theta']:.4f}      1.00000      0.00000\n")

            # Dados para o θ atual
            subset = df[df['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(
                    f"      {row['Phi']:.5f}      {row['Col1']:.2f}      {row['Col2']:.2f}      {row['Intensity']:.6f}\n")

            # Linha em branco para separar os blocos
            file.write("")


# Uso das funções
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df, num_theta, num_points, initial_theta = process_file(file_path)  # Lê os dados originais

# Aplicar a simetria vertical
df_with_symmetry = apply_vertical_symmetry(df)

# Salvar o DataFrame com simetria em formato de blocos
output_file = 'exp_Fe2_GaO_with_symmetry_blocks.txt'
save_to_txt_with_blocks(df_with_symmetry, output_file)

print(f"Arquivo salvo como: {output_file}")
