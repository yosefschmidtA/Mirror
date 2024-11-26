import numpy as np


def gauss1(X, p):
    """
    Função Gaussiana de um único pico.
    """
    xc, sigma, area = p
    return area * np.exp(-(X - xc) ** 2 / (2 * sigma ** 2))


def gaussn(X, Param):
    """
    Função que soma múltiplos picos gaussianos.
    """
    n = len(Param) // 3
    functot = np.zeros_like(X)

    for i in range(n):
        p = Param[3 * i:3 * (i + 1)]
        functot += gauss1(X, p)

    return functot

def read_lnls_v3(file_prefix, thetai, thetaf, dtheta, phii, phif, dphi, user_input):
    """
    Lê os dados dos arquivos conforme a estrutura do código IDL,
    e associa os dados conforme o valor da penúltima coluna (número de dados).
    """
    nphi = abs((phif - phii) // dphi + 1)
    ntheta = abs((thetaf - thetai) // dtheta + 1)

    thetatotal = np.zeros(ntheta * nphi)
    phitotal = np.zeros(ntheta * nphi)
    intenstotal = np.zeros(ntheta * nphi)

    idx = 0  # Contador para preencher os arrays

    # Inicializa os blocos
    blocks = []
    current_block = None

def read_lnls_v3(file_prefix, thetai, thetaf, dtheta, phii, phif, dphi, user_input):
    """
    Lê os dados dos arquivos e associa os dados de intensidade ao bloco escolhido pelo usuário.
    """
    # Inicializa as listas para armazenar os dados
    thetatotal = []
    phitotal = []
    intenstotal = []

    # Inicializa os blocos
    blocks = []
    current_block = None

    # Loop sobre os valores de theta e phi, montando os nomes dos arquivos
    for i in np.arange(thetai, thetaf + dtheta, dtheta):
        for j in np.arange(phii, phif + dphi, dphi):
            # Monta o nome do arquivo com base em theta e phi
            file_name = f"{file_prefix}{i}.{j}"

            try:
                print(f"Lendo o arquivo: {file_name}")  # Imprime o nome do arquivo
                with open(file_name, 'r') as f:
                    lines = f.readlines()

                    # Pular a primeira linha
                    lines = lines[1:]

                    # Processo para identificar e ler os blocos
                    for line in lines:
                        # Verifica se a linha contém 'LO' e pula se contiver
                        if 'LO' in line:
                            continue

                        parts = line.split()

                        if len(parts) == 7:  # Linha de cabeçalho
                            if current_block:  # Salva o bloco anterior
                                blocks.append(current_block)

                            initial_value = float(parts[0])  # Valor inicial (não utilizado)
                            step = float(parts[3])  # Passo (não utilizado, mas lido)
                            num_points = int(parts[5])  # Número de pontos de dados

                            current_block = {
                                'initial_value': initial_value,
                                'step': step,
                                'num_points': num_points,
                                'data': []
                            }
                        elif current_block:  # Adiciona dados ao bloco atual
                            try:
                                # Adiciona os valores de intensidade no bloco
                                current_block['data'].extend(map(float, parts))  # Considerando que os dados são float
                            except ValueError as e:
                                print(f"Erro ao processar a linha (valores não numéricos encontrados): {line}")
                                continue  # Ignora a linha com erro

                    if current_block:  # Adiciona o último bloco
                        blocks.append(current_block)

            except FileNotFoundError:
                print(f"Arquivo {file_name} não encontrado.")

    # Filtra o bloco correto com base no input do usuário
    selected_block = next((block for block in blocks if block['num_points'] == user_input), None)

    if not selected_block:
        print(f"Nenhum bloco encontrado com {user_input} pontos.")
        return thetatotal, phitotal, intenstotal

    # Extrai os dados corretamente
    data = selected_block['data']

    # Preenche os valores de theta, phi e intensidade
    idx = 0
    for i in range(len(data)):
        # Calcula theta e phi baseados nos parâmetros de entrada
        theta_value = thetai + (i // (phif // dphi)) * dtheta  # Calcula o valor de theta
        phi_value = phii + (i % (phif // dphi)) * dphi  # Calcula o valor de phi

        thetatotal.append(theta_value)
        phitotal.append(phi_value)
        intenstotal.append(data[i])

    print("Processamento concluído!")
    return thetatotal, phitotal, intenstotal



# Exemplo de chamada para a função
file_prefix = 'JL24_-'  # Prefixo dos arquivos
thetai = 12
thetaf = 69
dtheta = 3
phii = 0
phif = 357
dphi = 3

# Solicita o número de pontos para selecionar o bloco
user_input = 18

thetatotal, phitotal, intenstotal = read_lnls_v3(file_prefix, thetai, thetaf, dtheta, phii, phif, dphi, user_input)

# Imprimir os dados para verificar
print("thetatotal:", thetatotal)
print("phitotal:", phitotal)
print("intenstotal:", intenstotal)
