import os

# Parâmetros fornecidos
file_prefix = 'JL24_-'
thetai = 12
thetaf = 69
dtheta = 3
phii = 0
phif = 357
dphi = 3


# Função para gerar os nomes dos arquivos esperados
def generate_file_names(prefix, thetai, thetaf, dtheta, phii, phif, dphi):
    theta_values = [thetai + i * dtheta for i in range((thetaf - thetai) // dtheta + 1)]
    phi_values = [phii + j * dphi for j in range((phif - phii) // dphi + 1)]
    file_names = []
    for theta in theta_values:
        for phi in phi_values:
            file_name = f"{prefix}{theta}.{phi}"
            file_names.append(file_name)
    return file_names


# Gerar os nomes de arquivos esperados
expected_files = generate_file_names(file_prefix, thetai, thetaf, dtheta, phii, phif, dphi)

# Verificar quais arquivos existem no diretório atual
existing_files = [f for f in expected_files if os.path.isfile(f)]

# Listas para armazenar os dados separados
data_one_column = []
output_file_xps = "/home/yosef/PycharmProjects/Mirror/XPD-240724/Fe2p/saidaxps.txt"
with open(output_file_xps, 'w') as log_file:
    log_file.write("Lendo e processando os arquivos...\n")

    for file in existing_files:
        log_file.write(f"\n--- Processando arquivo: {file} ---\n")
        with open(file, 'r') as f:
            data_valid = False  # Flag para verificar se estamos no conjunto com 18 na sexta coluna
            for line in f:
                # Remove espaços e divide por colunas
                columns = line.strip().split()

                if len(columns) == 7:  # Linha com 7 colunas (cabeçalho)
                    # Verifique se a sexta coluna tem o valor 18
                    if float(columns[5]) == 18:  # Coluna 6 (índice 5)
                        data_valid = True  # Ativa o flag para processar os dados subsequentes
                        log_file.write(f"\nCabeçalho com 7 colunas encontrado: {columns}\n")

                # Se estivermos no conjunto válido, processa os dados
                elif data_valid and len(columns) == 1:  # Linha com 1 coluna
                    value = float(columns[0])
                    data_one_column.append(value)
                    log_file.write(f"{value}\n")


