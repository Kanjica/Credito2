from PIL import Image
import os

# Nomes dos seus arquivos originais
imagens = [
    "minhas_imagens/img1.png",
    "minhas_imagens/img2.png",
    "minhas_imagens/img3.png"
]

# Tamanhos ideais para o experimento (n = 10.000, 22.500 e 40.000)
tamanhos = [(100, 100), (150, 150), (200, 200)]

# Cria a pasta se não existir
pasta_saida = "./minhas_imagens"
os.makedirs(pasta_saida, exist_ok=True)

for i in range(3):
    arquivo_origem = imagens[i]
    novo_tamanho = tamanhos[i]
    
    try:
        img = Image.open(arquivo_origem).convert("RGB")
        img_redimensionada = img.resize(novo_tamanho, Image.Resampling.LANCZOS)
        
        # Salva na pasta do experimentador com um nome claro
        nome_saida = f"imagem_{novo_tamanho[0]}x{novo_tamanho[1]}.png"
        caminho_saida = os.path.join(pasta_saida, nome_saida)
        
        img_redimensionada.save(caminho_saida)
        print(f"Sucesso: {arquivo_origem} redimensionada para {novo_tamanho} e salva em {caminho_saida}")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{arquivo_origem}' não encontrado. Verifique se está na mesma pasta.")