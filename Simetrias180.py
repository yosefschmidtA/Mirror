import pandas as pd


# Função para processar o arquivo
def process_file(file_path):
    """
    Lê os dados do arquivo, coletando Phi, Col1, Col2 e Intensidade das colunas indicadas.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Inicializar uma lista para armazenar os dados
    data = []
    theta_value = None  # Para armazenar o valor de θ

    # Iterar sobre as linhas, começando após o cabeçalho (17 linhas)
    for i in range(17, len(lines)):
        line = lines[i].strip()  # Remove espaços em branco das extremidades

        if line:  # Verifica se a linha não está vazia
            parts = line.split()  # Divide a linha em partes

            if len(parts) == 6:  # Linha que contém θ
                theta_value = float(parts[3])  # Coleta θ na quarta posição

            elif len(parts) >= 4 and theta_value is not None:  # Linha que contém φ, colunas adicionais e intensidade
                phi = float(parts[0])  # Coleta φ na primeira posição
                col1 = float(parts[1])  # Coleta Col1 na segunda posição
                col2 = float(parts[2])  # Coleta Col2 na terceira posição
                intensity = float(parts[3])  # Coleta intensidade na quarta posição
                data.append([phi, col1, col2, theta_value, intensity])

    # Cria um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])
    return df


# Função para aplicar a simetria vertical nos dados de 0 a 180
def apply_symmetry_0_180(df):
    """
    Aplica a simetria vertical aos dados de Phi entre 0 e 180,
    incluindo as colunas adicionais Col1 e Col2.
    """
    # Filtrar dados de Phi entre 0 e 180
    df_original = df[df['Phi'] <= 180].copy()

    # Criar os dados simétricos para 180 a 360 (reflexão vertical)
    df_symmetry = df_original.copy()
    df_symmetry['Phi'] = 360 - df_original['Phi']  # Espelhar Phi verticalmente para 180 a 360

    # Marcar os valores extras (onde Phi == 360, que equivale a Phi == 0)
    df_symmetry['IsExtra'] = df_symmetry['Phi'] == 360
    df_original['IsExtra'] = False  # Dados originais não são extras

    # Combinar os dados originais e simétricos
    df_combined = pd.concat([df_original, df_symmetry], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Theta', 'Phi']).reset_index(drop=True)

    return df_combined


# Função para salvar os dados organizados em blocos
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
        file.write(f"     ({num_theta} {num_points} {theta_initial:.4f})\n")  # Cabeçalho dinâmico
        file.write(f"      {num_theta}     {num_points}       1       {num_theta}     {num_phi}     {num_points}\n")
        number_of_theta = 0

        # Loop para os diferentes valores de θ
        for theta in sorted(df['Theta'].unique()):
            number_of_theta += 1
            first_row = df[df['Theta'] == theta].iloc[0]
            file.write(
                f"       {number_of_theta}     {num_phi}       19.5900     {first_row['Theta']:.4f}      1.00000      0.00000\n"
            )

            # Selecionar apenas os dados para o θ atual
            subset = df[df['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(
                    f"      {row['Phi']:.5f}      {row['Col1']:.2f}      {row['Col2']:.2f}      {row['Intensity']:.6f}\n"
                )
            # Linha em branco para separar os blocos
            file.write("")


# Uso das funções
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df = process_file(file_path)  # Lê os dados originais

# Aplicar a simetria de 0 a 180
df_with_symmetry = apply_symmetry_0_180(df)

# Salvar o DataFrame com simetria em formato de blocos
output_file = 'exp_Fe2_GaO_with_symmetry_0_180_blocks.txt'
save_to_txt_with_blocks(df_with_symmetry, output_file)

print(f"Arquivo salvo como: {output_file}")
