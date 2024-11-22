import pandas as pd

# Função para aplicar a simetria vertical nos dados com intervalo definido pelo usuário
def apply_symmetry_custom_interval(df, start_angle, end_angle):
    """
    Aplica a simetria para um intervalo definido pelo usuário.
    Inclui Phi, Col1, Col2 e Intensity.
    """
    # Filtrar dados de Phi entre o intervalo fornecido
    df_original = df[(df['Phi'] >= start_angle) & (df['Phi'] <= end_angle)].copy()

    # Mapear Phi do intervalo fornecido para 0-180
    df_original['Phi'] = (df_original['Phi'] - start_angle) * (180 / (end_angle - start_angle))  # Remapear para 0-180

    # Ajustar para que Phi comece em 0
    df_original['Phi'] -= df_original['Phi'].iloc[0]

    # Criar os dados simétricos para 180 a 360 (espelhando o intervalo 0-180)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['Phi'] = 360 - df_original['Phi']  # Espelhar Phi para o intervalo 180 a 360

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Theta', 'Phi']).reset_index(drop=True)

    return df_combined

# Função para aplicar a segunda simetria, de 90 a 270
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
    with open(file_name, 'w') as file:
        for theta in sorted(df['Theta'].unique()):
            file.write(f"theta {theta:.2f}\n")  # Cabeçalho do bloco
            subset = df[df['Theta'] == theta]
            for _, row in subset.iterrows():
                file.write(f"{row['Phi']:.2f}\t{row['Col1']:.2f}\t{row['Col2']:.2f}\t{row['Intensity']:.6f}\n")
            file.write("\n")

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
