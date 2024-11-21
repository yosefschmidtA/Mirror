import pandas as pd

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

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)

    return df_combined

# Função para salvar os dados organizados em blocos
def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, onde cada bloco corresponde a um valor de θ (Theta).
    """
    with open(file_name, 'w') as file:
        for theta in sorted(df['Theta'].unique()):
            # Escrever o cabeçalho do bloco
            file.write(f"theta {theta:.2f}\n")
            # Selecionar apenas os dados para o θ atual
            subset = df[df['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(f"{row['Phi']:.2f}\t{row['Intensity']:.6f}\n")
            # Linha em branco para separar os blocos
            file.write("\n")

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

# Uso das funções
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df = process_file(file_path)  # Lê os dados originais

# Aplicar a simetria vertical
df_with_symmetry = apply_vertical_symmetry(df)

# Salvar o DataFrame com simetria em formato de blocos
output_file = 'exp_Fe2_GaO_with_symmetry_blocks.txt'
save_to_txt_with_blocks(df_with_symmetry, output_file)

print(f"Arquivo salvo como: {output_file}")
