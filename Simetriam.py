import pandas as pd

# Função para processar o arquivo
def process_file(file_path):
    """
    Lê os dados do arquivo e coleta Phi, Col1, Col2, Theta e Intensity.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inicializar uma lista para armazenar os dados
    data = []
    theta_value = None

    # Iterar sobre as linhas, começando após o cabeçalho (17 linhas)
    for i in range(17, len(lines)):
        line = lines[i].strip()

        if line:
            parts = line.split()

            if len(parts) == 6:  # Linha contendo Theta
                theta_value = float(parts[3])

            elif len(parts) >= 4 and theta_value is not None:  # Linha com os dados
                phi = float(parts[0])
                col1 = float(parts[1])
                col2 = float(parts[2])
                intensity = float(parts[3])
                data.append([phi, col1, col2, theta_value, intensity])

    # Criar um DataFrame com os dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])
    return df


# Função para aplicar a simetria com intervalo customizado
def apply_symmetry_custom_interval(df, start_angle, end_angle):
    """
    Aplica a simetria para um intervalo definido pelo usuário.
    Inclui Phi, Col1, Col2 e Intensity.
    """
    # Filtrar dados de Phi entre o intervalo fornecido
    df_original = df[(df['Phi'] >= start_angle) & (df['Phi'] <= end_angle)].copy()

    # Mapear Phi do intervalo fornecido para 0-180
    df_original['Phi'] = (df_original['Phi'] - start_angle) * (180 / (end_angle - start_angle))  # Remapear para 0-180

    # Criar os dados simétricos para 180 a 360 (espelhando o intervalo 0-180)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['Phi'] = 360 - df_original['Phi']  # Espelhar Phi para o intervalo 180 a 360

    # Marcar pontos extras (Phi == 360, equivalente a Phi == 0)
    df_symmetry_1['IsExtra'] = df_symmetry_1['Phi'] == 360
    df_original['IsExtra'] = False  # Dados originais não são extras

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Theta', 'Phi']).reset_index(drop=True)

    return df_combined


# Função para aplicar a segunda simetria
def apply_second_symmetry(df):
    """
    Aplica a segunda simetria aos dados entre 90 e 270 graus.
    Inclui Phi, Col1, Col2 e Intensity.
    """
    # Filtrar dados de Phi entre 90 e 270 para a segunda reflexão
    df_second_reflection = df[(df['Phi'] >= 90) & (df['Phi'] <= 270)].copy()

    # Mapear Phi de 90-270 para 0-180
    df_second_reflection['Phi'] -= 90

    # Criar a reflexão para o intervalo 180 a 360
    df_symmetry_2 = df_second_reflection.copy()
    df_symmetry_2['Phi'] = 360 - df_second_reflection['Phi']  # Espelhar Phi para o intervalo 180 a 360

    # Marcar pontos extras (Phi == 360, equivalente a Phi == 0)
    df_symmetry_2['IsExtra'] = df_symmetry_2['Phi'] == 360
    df_second_reflection['IsExtra'] = False  # Dados originais não são extras

    # Combinar os dados
    df_combined_second_reflection = pd.concat([df_second_reflection, df_symmetry_2], ignore_index=True)

    # Remover duplicatas
    df_combined_second_reflection = df_combined_second_reflection.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar os dados
    df_combined_second_reflection = df_combined_second_reflection.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)

    return df_combined_second_reflection


# Função para salvar os dados organizados em blocos
def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, incluindo Phi, Col1, Col2 e Intensity.
    """
    # Filtrar apenas os pontos que não são extras para cálculos
    valid_df = df[~df['IsExtra']]  # Considerar apenas os valores válidos
    num_theta = valid_df['Theta'].nunique()  # Número de ângulos theta únicos
    num_phi = valid_df['Phi'].nunique()  # Número de ângulos phi únicos
    num_points = num_theta * num_phi  # Total de pontos
    theta_initial = df['Theta'].min()  # Valor inicial de Theta

    with open(file_name, 'w') as file:
        # Cabeçalho inicial
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
        file.write(f"     ({num_theta} {num_points} {theta_initial:.4f})\n")
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")

        # Loop para os diferentes valores de θ
        number_of_theta = 0
        for theta in sorted(df['Theta'].unique()):
            number_of_theta += 1
            first_row = df[df['Theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {first_row['Theta']:.4f}      1.00000      0.00000\n"
            )
            subset = df[df['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(
                    f"      {row['Phi']:.5f}      {row['Col1']:.2f}      {row['Col2']:.2f}      {row['Intensity']:.6f}\n"
                )
            file.write("")


# Uso das funções
file_path = 'exp_Fe2_GaO.out'
df = process_file(file_path)

# Solicitar intervalo de ângulos ao usuário
start_angle = float(input("Digite o ângulo inicial: "))
end_angle = float(input("Digite o ângulo final: "))

# Aplicar simetria com o intervalo fornecido
df_with_symmetry = apply_symmetry_custom_interval(df, start_angle, end_angle)

# Aplicar segunda reflexão no intervalo de 90 a 270
df_with_second_reflection = apply_second_symmetry(df_with_symmetry)

# Salvar o DataFrame com simetrias em formato de blocos
output_file = 'exp_Fe2_GaO_with_symmetry_blocks.txt'
save_to_txt_with_blocks(df_with_second_reflection, output_file)

print(f"Arquivo salvo como: {output_file}")
