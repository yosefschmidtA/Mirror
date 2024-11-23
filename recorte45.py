import pandas as pd


# Função para processar o arquivo
def process_file(file_path):
    """
    Lê os dados do arquivo, coletando Phi, Col1, Col2, Theta e Intensidade.
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

            elif len(parts) >= 4 and theta_value is not None:  # Linha que contém φ e intensidade
                phi = float(parts[0])  # Coleta φ na primeira posição
                col1 = float(parts[1])  # Coleta Col1 na segunda posição
                col2 = float(parts[2])  # Coleta Col2 na terceira posição
                intensity = float(parts[3])  # Coleta intensidade na quarta posição
                data.append([phi, col1, col2, theta_value, intensity])

    # Cria um DataFrame a partir dos dados coletados
    df = pd.DataFrame(data, columns=['Phi', 'Col1', 'Col2', 'Theta', 'Intensity'])
    return df


# Função para filtrar dados de 0 a 45 graus
def filter_data_0_45(df):
    """
    Filtra os dados para incluir apenas valores de Phi entre 0 e 45 graus.
    """
    df_filtered = df[(df['Phi'] >= 0) & (df['Phi'] <= 45)].copy()
    return df_filtered


# Função para salvar os dados organizados em blocos
def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, limitados a Phi de 0 a 45 graus.
    """
    num_theta = df['Theta'].nunique()  # Número de ângulos theta únicos
    num_phi = df['Phi'].nunique()  # Número de ângulos phi únicos
    num_points = len(df)  # Total de pontos

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
        file.write(f"     ({num_theta} {num_points} {df['Theta'].min():.4f})\n")  # Cabeçalho dinâmico
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
            file.write("\n")


# Uso das funções
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do arquivo
df = process_file(file_path)  # Lê os dados originais

# Filtrar os dados de 0 a 45 graus
df_filtered = filter_data_0_45(df)

# Salvar apenas os dados de 0 a 45 graus
output_file = 'exp_Fe2_GaO_filtered_0_45.txt'
save_to_txt_with_blocks(df_filtered, output_file)

print(f"Arquivo salvo com os dados de 0 a 45 graus: {output_file}")
