import pandas as pd

def apply_symmetry_custom_interval(df, start_angle, end_angle):
    # Verificar os nomes das colunas
    print("Colunas disponíveis:", df.columns)

    # Certificar-se de que estamos utilizando o nome correto da coluna 'phi'
    if 'phi' not in df.columns:
        print("Erro: A coluna 'phi' não foi encontrada.")
        return df  # Se a coluna não existir, retorna o DataFrame original

    # Filtrar dados de phi entre o intervalo fornecido
    df_original = df[(df['phi'] >= start_angle) & (df['phi'] <= end_angle)].copy()

    # Mapear phi do intervalo fornecido para 0-180
    df_original['phi'] = (df_original['phi'] - start_angle) * (180 / (end_angle - start_angle))

    # Garantir que phi começa em 0
    df_original['phi'] = df_original['phi'] - df_original['phi'].iloc[0]

    # Ajuste para garantir que o valor 180 seja corretamente incluído
    df_original['phi'] = df_original['phi'].round(2)

    # Criar os dados simétricos para 180 a 360 (espelhando o intervalo 0-180)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['phi'] = 360 - df_original['phi']

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['phi', 'theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['phi', 'theta']).reset_index(drop=True)

    return df_combined

def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, onde cada bloco corresponde a um valor de θ (theta).
    """
    with open(file_name, 'w') as file:
        for theta in sorted(df['theta'].unique()):
            # Escrever o cabeçalho do bloco
            file.write(f"theta {theta:.2f}\n")
            # Selecionar apenas os dados para o θ atual
            subset = df[df['theta'] == theta]
            for _, row in subset.iterrows():
                file.write(f"{row['phi']:.2f}\t{row['intensity']:.6f}\n")
            # Linha em branco para separar os blocos
            file.write("\n")

def read_generic_data(file_path):
    """
    Lê os dados de um arquivo e os organiza em um DataFrame.
    """
    data = []
    current_theta = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Verificar se a linha contém um valor de 'theta'
            if line.startswith('theta'):
                current_theta = float(line.split()[1])

            # Verificar se a linha contém valores de 'phi' e 'intensity'
            elif line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 2:
                    phi = float(parts[0])
                    intensity = float(parts[1])
                    data.append({'theta': current_theta, 'phi': phi, 'intensity': intensity})

    return pd.DataFrame(data)

# Uso das funções
file_path = 'exp_Fe2_GaO.out'
df = read_generic_data(file_path)

print(f"Colunas após leitura do arquivo: {df.columns}")

# Solicitar intervalo de ângulos ao usuário
start_angle = float(input("Digite o ângulo inicial: "))
end_angle = float(input("Digite o ângulo final: "))

# Aplicar a simetria com o intervalo fornecido
df_with_symmetry = apply_symmetry_custom_interval(df, start_angle, end_angle)

# Salvar o DataFrame com simetria em formato de blocos
output_file = 'exp_Fe2_GaO_with_symmetry_blocks.txt'
save_to_txt_with_blocks(df_with_symmetry, output_file)

print(f"Arquivo salvo como: {output_file}")
