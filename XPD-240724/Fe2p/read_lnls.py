import os
import numpy as np
from scipy.integrate import trapezoid
# Parâmetros fornecidos
file_prefix = 'JL24_-'
thetai = 12
thetaf = 69
dtheta = 3
phii = 0
phif = 357
dphi = 3
channel = 18

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

    for file in existing_files:

        try:
            # Assumindo que o formato do arquivo é 'prefixo_theta.phi'
            theta, phi = file[len(file_prefix):].split('.')
            theta = int(theta)
            phi = int(phi)
        except ValueError:
            continue  # Caso o nome do arquivo não tenha o formato esperado

        with open(file, 'r') as f:
            data_valid = False  # Flag para verificar se estamos no conjunto com 18 na sexta coluna
            contador_banda = 0
            for line in f:
                # Remove espaços e divide por colunas
                columns = line.strip().split()

                if len(columns) == 7:  # Linha com 7 colunas (cabeçalho)
                    # Verifique se a sexta coluna tem o valor 18
                    if float(columns[5]) == channel:  # Coluna 6 (índice 5)
                        data_valid = True  # Ativa o flag para processar os dados subsequentes

                        log_file.write(f"\n{theta} {phi} {channel}\n")

                # Se estivermos no conjunto válido, processa os dados
                elif data_valid and len(columns) == 1:  # Linha com 1 coluna
                    value = float(columns[0])
                    contador_banda += 1
                    data_one_column.append(value)

                    log_file.write(f"{value} {contador_banda}\n")


def shirley_background(x_data, y_data, init_back, end_back, n_iterations=6):
    """
    Calcula o fundo de Shirley para um espectro de intensidade.

    Parameters:
    x_data (array): O vetor de energias de ligação (ou qualquer outra variável x).
    y_data (array): O vetor de intensidades do espectro.
    init_back (int): Índice inicial para calcular o fundo.
    end_back (int): Índice final para calcular o fundo.
    n_iterations (int): Número de iterações para refinar o fundo.

    Returns:
    background (array): O vetor de fundo calculado.
    """
    # Inicializa o vetor de fundo com os valores nas extremidades
    background = np.zeros_like(y_data)
    background0 = np.zeros_like(y_data)

    # Definindo os valores de intensidade nas extremidades
    a = y_data[init_back]
    b = y_data[end_back]

    # Calcula o fundo de Shirley por n iterações
    for nint in range(n_iterations):
        for k2 in range(end_back, init_back - 1, -1):
            sum1 = 0
            sum2 = 0
            for k in range(end_back, k2 - 1, -1):
                sum1 += y_data[k] - background0[k]
            for k in range(end_back, init_back - 1, -1):
                sum2 += y_data[k] - background0[k]

            # Calcula o fundo interpolado entre as extremidades
            background[k2] = (a - b) * sum1 / sum2 + b

        # Ajuste o fundo para as extremidades
        background[:init_back] = background[init_back]
        background[end_back:] = background[end_back]

        # Atualiza o fundo de referência para a próxima iteração
        background0 = background.copy()

    return background


def process_file(file_name, output_file):
    """
    Lê o arquivo de entrada e processa os dados aplicando o fundo Shirley.
    """
    with open(file_name, 'r') as file:
        data = file.readlines()

    corrected_data = []
    block = []
    header = None

    # Processa cada linha do arquivo
    for line in data:
        columns = line.strip().split()
        if len(columns) == 3:
            # Cabeçalho do bloco
            if block:  # Se houver um bloco acumulado, processa
                corrected_data.append((header, process_block(block)))
                block = []
            header = line.strip()  # Salva o cabeçalho do bloco atual
        elif len(columns) == 2:
            # Adiciona valores ao bloco atual
            block.append([float(columns[0]), int(columns[1])])

    # Processa o último bloco
    if block:
        corrected_data.append((header, process_block(block)))

    # Grava os resultados corrigidos no arquivo de saída
    with open(output_file, 'w') as out_file:
        for header, block_data in corrected_data:
            # Agora, extraímos a área total do segundo valor retornado por process_block
            _, total_area = block_data
            out_file.write(f"{header} {total_area:.1f}\n")
            for y_corr, x in block_data[0]:  # Escreve os dados corrigidos
                out_file.write(f"{y_corr:.2f} {x}\n")


def process_block(block):
    """
    Processa um bloco de 18 pontos aplicando o fundo Shirley.
    """
    y_values = np.array([row[0] for row in block])
    x_values = np.array([row[1] for row in block])

    # Define os índices iniciais e finais do fundo
    initback = 0
    endback = len(x_values) - 1

    # Calcula o fundo Shirley
    bg = shirley_background(x_values, y_values, initback, endback)

    # Corrige os valores de intensidade
    y_corrected = y_values - bg
    total_area = trapezoid(np.abs(y_corrected), x_values)

    # Imprimir a área total
    print(f'Área total corrigida: {total_area}')
    return list(zip(y_corrected, x_values)), total_area


# Arquivos de entrada e saída
file_name = "saidaxps.txt"
output_file = "saidashirley.txt"

# Processa o arquivo
process_file(file_name, output_file)

