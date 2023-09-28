import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

imagemCoins = cv2.imread("pt_03/coins.png")
imagemLenaRuido = cv2.imread("pt_03/lena_ruido.jpeg")
imagemMargaridas = cv2.imread("pt_03/margaridas.png")

# Com a imagem lena_ruido.png aplique o filtro da Mediana
def filtroMediana():
    tamanho_kernel = 5 
    imagem_filtrada = cv2.medianBlur(imagemLenaRuido, tamanho_kernel)

    cv2.imshow('Imagem Original', imagemLenaRuido)
    cv2.imshow('Imagem Com Filtro de Mediana', imagem_filtrada)

    cv2.waitKey(0)

# Com a coins.png aplique os filtros Sobel e Prewitt
def filtroSobelPrewitt():
    # Filtro Sobel
    sobel_x = cv2.Sobel(imagemCoins, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagemCoins, cv2.CV_64F, 0, 1, ksize=3)

    imagemFiltroSobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Filtro Prewitt
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    prewitt_x = cv2.filter2D(imagemCoins, -1, kernel_x)
    prewitt_y = cv2.filter2D(imagemCoins, -1, kernel_y)

    imagemFiltroPrewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    fig, ax = plt.subplots(1, 5)
    ax[0].imshow(imagemCoins)
    ax[0].set_title('Imagem Original')

    ax[1].imshow(sobel_x)
    ax[1].set_title('Filtro Sobel - x')
    ax[2].imshow(sobel_y)
    ax[2].set_title('Filtro Sobel - y')

    ax[3].imshow(prewitt_x)
    ax[3].set_title('Filtro Prewitt - x')
    ax[4].imshow(prewitt_y)
    ax[4].set_title('Filtro Prewitt - y')
    plt.show()

# Com a margaridas.png para teste, crie um script que permita particionar uma imagem 
# em diferentes quadrados e aplique um método de binarização em cada uma das partes.
def particionarBinarizacao():
    num_linhas = 4
    num_colunas = 4

    altura, largura, _ = imagemMargaridas.shape
    altura_quadrado = altura // num_linhas
    largura_quadrado = largura // num_colunas

    for linha in range(num_linhas):
        for coluna in range(num_colunas):
            y1 = linha * altura_quadrado
            y2 = (linha + 1) * altura_quadrado
            x1 = coluna * largura_quadrado
            x2 = (coluna + 1) * largura_quadrado

            quadrado = imagemMargaridas[y1:y2, x1:x2]

            quadrado_em_escala_de_cinza = cv2.cvtColor(quadrado, cv2.COLOR_BGR2GRAY)

            _, binarizado = cv2.threshold(quadrado_em_escala_de_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            imagemMargaridas[y1:y2, x1:x2] = cv2.cvtColor(binarizado, cv2.COLOR_GRAY2BGR)

   
    cv2.imshow('Imagem com Quadrados Binarizados', imagemMargaridas)
    cv2.waitKey(0)