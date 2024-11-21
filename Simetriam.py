import pandas as pd

# Função para aplicar a simetria vertical nos dados com intervalo definido pelo usuário
def apply_symmetry_custom_interval(df, start_angle, end_angle):
    # Filtrar dados de Phi entre o intervalo fornecido
    df_original = df[(df['Phi'] >= start_angle) & (df['Phi'] <= end_angle)].copy()

    # Mapear Phi do intervalo fornecido para 0-180
    df_original['Phi'] = (df_original['Phi'] - start_angle) * (180 / (end_angle - start_angle))  # Remapear para 0-180

    # Garantir que Phi começa em 0
    df_original['Phi'] = df_original['Phi'] - df_original['Phi'].iloc[0]  # Ajustar para que o primeiro valor de Phi seja 0

    # Ajuste para garantir que o valor 180 seja corretamente incluído
    df_original['Phi'] = df_original['Phi'].round(2)  # Arredondar para dois decimais para garantir precisão

    # Criar os dados simétricos para 180 a 360 (espelhando o intervalo 0-180)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['Phi'] = 360 - df_original['Phi']  # Espelhar Phi para o intervalo 180 a 360

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['Phi', 'Theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['Phi', 'Theta']).reset_index(drop=True)

    return df_combined

# Função para aplicar a segunda simetria, de 90 a 270, após a primeira reflexão
def apply_second_symmetry(df):
    # Filtrar dados de Phi entre 90 e 270 para a segunda reflexão
    df_second_reflection = df[(df['Phi'] >= 90) & (df['Phi'] <= 270)].copy()

    # Mapear Phi de 90-270 para 0-180
    df_second_reflection['Phi'] = (df_second_reflection['Phi'] - 90) * (180 / 180)  # Remapear para 0-180

    # Garantir que Phi começa em 0
    df_second_reflection['Phi'] = df_second_reflection['Phi'] - df_second_reflection['Phi'].iloc[0]  # Ajuste

    # Criar a reflexão para o intervalo 180 a 360
    df_symmetry_2 = df_second_reflection.copy()
    df_symmetry_2['Phi'] = 360 - df_second_reflection['Phi']  # Espelhar

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

# Solicitar intervalo de ângulos ao usuário
start_angle = float(input("Digite o ângulo inicial: "))
end_angle = float(input("Digite o ângulo final: "))

# Aplicar a simetria com o intervalo fornecido
df_with_symmetry = apply_symmetry_custom_interval(df, start_angle, end_angle)

# Aplicar a segunda reflexão no intervalo de 90 a 270
df_with_second_reflection = apply_second_symmetry(df_with_symmetry)

# Salvar o DataFrame com as simetrias em formato de blocos
output_file = 'exp_Fe2_GaO_with_second_reflection_blocks.txt'
save_to_txt_with_blocks(df_with_second_reflection, output_file)

print(f"Arquivo salvo como: {output_file}")