
# Leitor de Documentos com OCR e Áudio

![Status](https://img.shields.io/badge/status-BETA-yellow) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![Plataforma](https://img.shields.io/badge/platform-Windows-lightgrey)

Este projeto é uma ferramenta de acessibilidade projetada para ler documentos físicos em voz alta. Utilizando uma webcam ou uma imagem estática, o programa detecta o conteúdo de uma página, realiza o Reconhecimento Óptico de Caracteres (OCR) para extrair o texto e o converte em áudio.

O objetivo principal é auxiliar pessoas com deficiência visual ou dificuldades de leitura, transformando texto impresso em uma experiência auditiva.

## ⚠️ Status: Versão BETA

Este software está em fase de desenvolvimento **BETA**. Atualmente, seu funcionamento é otimizado para **papel digitalizado** sob condições específicas. A detecção da página depende de uma estrutura visual pré-definida.

![image](https://github.com/user-attachments/assets/25674bf8-2e41-4a20-b1f5-28f1a1a7eca7)
---

## ✨ Funcionalidades Principais

* **Detecção de Página:** Identifica a área de texto em uma imagem usando quatro quadrados pretos como âncoras.
* **Correção de Perspectiva:** Alinha e "aplana" a imagem da página para melhorar a precisão do OCR.
* **Pré-processamento de Imagem:** Aplica uma série de filtros configuráveis para otimizar a imagem para o reconhecimento de texto.
* **OCR com Tesseract:** Extrai o texto da imagem processada usando o motor Tesseract.
* **Síntese de Voz (TTS):** Lê o texto extraído em voz alta usando a biblioteca `pyttsx3`.
* **Controle por Palavras-Gatilho:** Utiliza palavras-chave específicas no documento para controlar a leitura (iniciar, parar, pular página, etc.).
* **Interface Gráfica Simples:** Oferece visualização da câmera/imagem, das etapas de processamento (em modo debug) e do texto reconhecido.

## 📄 Pré-requisito Essencial: A Estrutura da Página

Para que o programa funcione corretamente nesta fase, a página a ser lida **precisa** ser estruturada com **quatro quadrados pretos sólidos**, um em cada canto. Estes quadrados servem como marcadores para que o `OpenCV` possa detectar a área exata do texto, isolá-la e corrigir sua perspectiva.

## 🔊 Palavras-Gatilho para Controle de Áudio

Para guiar a leitura, o sistema reconhece "palavras-gatilho" específicas que devem estar presentes no texto do documento.

* `TITULO:`: Ao encontrar esta palavra, o programa anuncia "Título do Texto" e lê o texto que a segue na mesma linha.
* `INICIO:`: Indica o começo do conteúdo principal. O programa anuncia "Início do texto" e começa a ler todas as linhas subsequentes.
* `PULA_PAGINA`: Comando para finalizar a leitura da página atual. O programa avisa ao usuário para virar a página.
* `FINAL`: Indica que o documento inteiro terminou. O programa anuncia "Fim do Texto" e encerra o processo de leitura.

### Exemplo de Texto na Página:

TITULO: A História da Computação

INICIO: A computação tem uma história rica e fascinante.
As primeiras máquinas eram mecânicas e enormes.
Com o tempo, a invenção do transistor revolucionou tudo.

PULA_PAGINA

## 🛠️ Requisitos e Instalação

Siga os passos abaixo para configurar o ambiente e executar o projeto.

### 1. Tesseract-OCR

Este projeto depende do motor Tesseract.

* **Windows:** Baixe e instale o Tesseract através do [instalador oficial](https://github.com/UB-Mannheim/tesseract/wiki).
    * **Importante:** Durante a instalação, certifique-se de marcar a opção para adicionar o Tesseract ao `PATH` do sistema.
    * Selecione o pacote de idioma **"Portuguese"** para que o reconhecimento em português (`por`) funcione.

### 2. Dependências Python

As bibliotecas Python necessárias podem ser instaladas via `pip`. É recomendado criar um ambiente virtual.

pip install opencv-python numpy pyttsx3 Pillow

### 3. Código do Projeto

Clone ou baixe o repositório para a sua máquina local.

### 4. Como Usar

  `Para usar a Webcam:` Certifique-se de que a variável CAMINHO_IMAGEM_TESTE no topo do script esteja comentada ou definida como None.
  `Para usar uma Imagem de Teste:` Descomente a variável CAMINHO_IMAGEM_TESTE e aponte para um arquivo de imagem válido.

### OBSERVAÇÃO:

O script possui uma seção de constantes globais no início do arquivo. Usuários avançados podem alterar esses valores para ajustar a sensibilidade da detecção e a qualidade do pré-processamento de imagem, melhorando a precisão do OCR em diferentes condições de iluminação e qualidade de câmera.

### Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

`"Seja a mudança que você quer ver no mundo." – Mahatma Gandhi`
