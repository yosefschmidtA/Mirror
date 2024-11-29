def somar_coluna(file_path):
    # Inicializa a soma da segunda coluna
    soma_coluna_2 = 0.0

    # Abre o arquivo para leitura
    with open(file_path, 'r') as file:
        # Lê cada linha do arquivo
        for line in file:
            # Divide a linha em campos (valores separados por espaços)
            valores = line.split()

            # Verifica se a linha contém pelo menos 2 valores (para evitar erros)
            if len(valores) >= 2:
                # Adiciona o valor da segunda coluna (valores[1]) à soma
                soma_coluna_2 += float(valores[1])

    return soma_coluna_2


# Caminho do arquivo
file_path = 'shirley.txt'

# Chama a função e exibe o resultado
soma = somar_coluna(file_path)
print(f'A soma dos valores da segunda coluna é: {soma}')
