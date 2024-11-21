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
    df_original['phi'] = (df_original['phi'] - start_angle) * (180 / (end_angle - start_angle))  # Remapear para 0-180

    # Garantir que phi começa em 0
    df_original['phi'] = df_original['phi'] - df_original['phi'].iloc[0]  # Ajustar para que o primeiro valor de phi seja 0

    # Ajuste para garantir que o valor 180 seja corretamente incluído
    df_original['phi'] = df_original['phi'].round(2)  # Arredondar para dois decimais para garantir precisão

    # Criar os dados simétricos para 180 a 360 (espelhando o intervalo 0-180)
    df_symmetry_1 = df_original.copy()
    df_symmetry_1['phi'] = 360 - df_original['phi']  # Espelhar phi para o intervalo 180 a 360

    # Combinar todos os dados (originais e simétricos)
    df_combined = pd.concat([df_original, df_symmetry_1], ignore_index=True)

    # Remover duplicatas
    df_combined = df_combined.drop_duplicates(subset=['phi', 'theta']).reset_index(drop=True)

    # Ordenar para manter consistência
    df_combined = df_combined.sort_values(by=['phi', 'theta']).reset_index(drop=True)

    return df_combined

# Função para salvar os dados organizados em blocos
def save_to_txt_with_blocks(df, file_name):
    """
    Salva os dados organizados em blocos, onde cada bloco corresponde a um valor de θ (Theta).
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

# Função para ler os dados de forma genérica
def read_generic_data(file_path):
    # Inicializar lista para armazenar os dados
    data = []
    current_theta = None

    with open(file_path, 'r') as file:
        # Ler o arquivo linha por linha
        for line in file:
            # Remover espaços extras e quebras de linha
            line = line.strip()

            # Verificar se a linha contém um valor de 'theta'
            if line.startswith('theta'):
                current_theta = float(line.split()[1])  # Pega o valor de theta após a palavra 'theta'

            # Verificar se a linha contém um par de valores (phi, intensidade)
            elif line and not line.startswith('#'):  # Ignora linhas vazias ou comentários
                parts = line.split()
                if len(parts) == 2:  # Verifica se é um par de valores
                    phi = float(parts[0])
                    intensity = float(parts[1])
                    # Adiciona os dados com o valor de 'theta' associado
                    data.append({'theta': current_theta, 'phi': phi, 'intensity': intensity})

    # Converter os dados para um DataFrame
    df = pd.DataFrame(data)
    return df

# Uso das funções
file_path = 'exp_Fe2_GaO.out'  # Substitua pelo caminho do seu arquivo
df = read_generic_data(file_path)  # Lê os dados originais

# Verifique as colunas para garantir que estão corretas
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
