import cv2
import numpy as np
import pyttsx3
import os
import time
from PIL import Image, ImageTk, ImageEnhance
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import traceback
import queue 

# --- Configurações e Constantes Globais ---
DEBUG_VISUAL = True 

CAMINHO_IMAGEM_TESTE = r"C:\Users\Maycon JD\Pictures\id_imagem\teste_print.png"
# CAMINHO_IMAGEM_TESTE = None # Descomente para usar a webcam

TEMP_DIR = os.environ.get('TEMP', os.path.join(os.getcwd(), 'temp_ocr_v2'))
if not os.path.exists(TEMP_DIR):
    try: os.makedirs(TEMP_DIR)
    except OSError as e: print(f"Erro ao criar TEMP_DIR: {e}"); TEMP_DIR = os.getcwd()

ROI_PROCESSADA_PATH = os.path.join(TEMP_DIR, "ocr_input_image.png")

MIN_QUADRADO_AREA_DETECT = 300; MAX_QUADRADO_AREA_DETECT = 50000   
PROPORCAO_QUADRADO_MIN_DETECT = 0.70; PROPORCAO_QUADRADO_MAX_DETECT = 1.30 
GAUSSIAN_BLUR_KERNEL_DETECCAO = (5, 5) 
ADAPTIVE_THRESH_BLOCK_SIZE_DETECCAO = 35 # Ajustado para o último teste
ADAPTIVE_THRESH_C_DETECCAO = 9           # Ajustado para o último teste
APPROX_POLY_EPSILON_DETECT = 0.035    

APLICAR_MEDIAN_BLUR_OCR = True      # Mude para True para testar, False para desativar
MEDIAN_BLUR_KERNEL_SIZE_OCR = 3     # Tamanho do kernel para Median Blur (deve ser ímpar, ex: 3 ou 5)

APLICAR_NITIDEZ_OCR = False          # Mude para True para testar filtro de nitidez
ADAPTIVE_THRESH_BLOCK_SIZE_OCR = 41  # Ímpar. Experimente 25, 35, 41, 51... 
ADAPTIVE_THRESH_C_OCR = 11           # Inteiro (ex: 5, 7, 9, 11, 13, 15...)
INVERTER_BINARIZACAO_FINAL_OCR = False # True se o texto estiver branco em fundo preto após binarização
APLICAR_MORFOLOGIA_OPEN_OCR = False  # True para remover pequenos ruídos
APLICAR_MORFOLOGIA_CLOSE_OCR = False # True para fechar pequenos buracos no texto
KERNEL_MORFOLOGIA_OCR_SIZE = (2,2) 

IMG_LARGURA_GUI_PRINCIPAL = 500; IMG_ALTURA_GUI_PRINCIPAL = 400
IMG_LARGURA_GUI_DEBUG = 320; IMG_ALTURA_GUI_DEBUG = 240
IMG_LARGURA_GUI_ROI_PROC = 400; IMG_ALTURA_GUI_ROI_PROC = 500
ROI_LARGURA_SAIDA_PARA_OCR = 1200; ROI_ALTURA_SAIDA_PARA_OCR = 1700 
TESSERACT_IDIOMA = 'por'
TESSERACT_PSM_MODE = '3' 

# --- MARCADORES DE TEXTO ATUALIZADOS ---
MARCADORES_AUDIO = {
    "TITULO:": "Título do Texto",          # Com colon
    "INICIO:": "Início do texto",          # Com colon
    "PULA_PAGINA": "Fim da página, por favor, passe para a próxima", # Sem colon
    "FINAL": "Fim do Texto"                # Sem colon
}

# (Inicialização do TTS e funções falar_texto_windows, order_points como antes)
# ... (código omitido para brevidade, mas é o mesmo da resposta anterior) ...
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 180)
    voices = tts_engine.getProperty('voices')
    for voice in voices:
        if "brazil" in voice.name.lower() or "portuguese" in voice.name.lower() or \
           "luciana" in voice.name.lower() or "maria" in voice.name.lower() or \
           (hasattr(voice, 'languages') and voice.languages and 'pt_BR' in voice.languages[0]):
            tts_engine.setProperty('voice', voice.id); print(f"Voz TTS: {voice.name}"); break
except Exception as e: print(f"Erro TTS: {e}"); tts_engine = None

tts_queue = queue.Queue()
tts_thread_stop_event = threading.Event()
tts_worker_thread = None

def tts_worker_function():
    while not tts_thread_stop_event.is_set():
        try:
            texto_para_falar = tts_queue.get(block=True, timeout=0.2)
            if texto_para_falar is None: tts_thread_stop_event.set(); break
            if tts_engine:
                # print(f"TTS Worker: Falando '{texto_para_falar}'") # Log opcional
                tts_engine.say(texto_para_falar)
                tts_engine.runAndWait()
            tts_queue.task_done()
        except queue.Empty: continue
        except Exception as e:
            print(f"Erro no TTS worker: {e}")
            if not tts_queue.empty(): tts_queue.task_done()
    print("TTS Worker finalizado.")

if tts_engine:
    tts_worker_thread = threading.Thread(target=tts_worker_function, daemon=True)
    tts_worker_thread.start()

def falar_texto_windows(texto):
    if tts_engine:
        print(f"AUDIO (enfileirado): {texto}")
        tts_queue.put(texto)
    else:
        print(f"TTS indisponível. Simulado: {texto}")

def order_points(pts_array_numpy):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_array_numpy.sum(axis=1); rect[0] = pts_array_numpy[np.argmin(s)]; rect[2] = pts_array_numpy[np.argmax(s)]
    diff = np.diff(pts_array_numpy, axis=1); rect[1] = pts_array_numpy[np.argmin(diff)]; rect[3] = pts_array_numpy[np.argmax(diff)]
    return rect

def detectar_delimitadores_pagina(imagem_cv_original, app_ref):
    # (Implementação como na resposta anterior)
    # ...
    gray = cv2.cvtColor(imagem_cv_original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_DETECCAO, 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 
                                   ADAPTIVE_THRESH_BLOCK_SIZE_DETECCAO, 
                                   ADAPTIVE_THRESH_C_DETECCAO)
    if DEBUG_VISUAL and app_ref and hasattr(app_ref, 'lbl_img_thresh_debug'):
        app_ref.exibir_imagem_em_label(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 
                                       app_ref.lbl_img_thresh_debug, 
                                       IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrados_candidatos = []
    img_com_contornos_debug = imagem_cv_original.copy()
    # print("\n--- Depurando Detecção de Quadrados ---") 
    # (Os prints detalhados podem ser reativados se necessário)
    for i, cnt in enumerate(contornos):
        area = cv2.contourArea(cnt)
        if MIN_QUADRADO_AREA_DETECT < area < MAX_QUADRADO_AREA_DETECT:
            perimetro = cv2.arcLength(cnt, True)
            aprox_poligono = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON_DETECT * perimetro, True)
            if len(aprox_poligono) == 4 and cv2.isContourConvex(aprox_poligono):
                (x, y, w, h) = cv2.boundingRect(aprox_poligono)
                aspect_ratio = w / float(h) if h != 0 else 0
                if PROPORCAO_QUADRADO_MIN_DETECT <= aspect_ratio <= PROPORCAO_QUADRADO_MAX_DETECT:
                    M = cv2.moments(aprox_poligono)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                        quadrados_candidatos.append({'centroid': np.array([cX, cY]), 'contour': aprox_poligono})
                        cv2.drawContours(img_com_contornos_debug, [aprox_poligono], -1, (0, 255, 0), 2)
                        cv2.circle(img_com_contornos_debug, (cX, cY), 5, (0, 0, 255), -1)
    if DEBUG_VISUAL and app_ref and hasattr(app_ref, 'lbl_img_contornos_debug'):
        app_ref.exibir_imagem_em_label(img_com_contornos_debug, app_ref.lbl_img_contornos_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
    if len(quadrados_candidatos) == 4:
        quadrados_centroids = [qc['centroid'] for qc in quadrados_candidatos]; pontos_np = np.array(quadrados_centroids)
        pontos_ordenados = order_points(pontos_np)
        img_gui_com_deteccoes_finais = imagem_cv_original.copy()
        for qc in quadrados_candidatos: cv2.drawContours(img_gui_com_deteccoes_finais, [qc['contour']], -1, (0,255,0),2)
        cores_pontos_ordenados = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]; nomes_pontos_ordenados = ["SE", "SD", "ID", "IE"]
        for idx, pt_ord in enumerate(pontos_ordenados):
            cv2.circle(img_gui_com_deteccoes_finais, tuple(pt_ord.astype(int)), 10, cores_pontos_ordenados[idx], -1)
            cv2.putText(img_gui_com_deteccoes_finais, nomes_pontos_ordenados[idx], tuple((pt_ord - [15,15]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        print("--- 4 quadrados delimitadores encontrados! ---")
        return pontos_ordenados, img_gui_com_deteccoes_finais
    else:
        print(f"--- Detectados {len(quadrados_candidatos)} quadrados válidos. Esperava 4. ---")
        return None, img_com_contornos_debug

def corrigir_perspectiva(imagem_cv_original, pontos_pagina_ordenados_np):
    # (Implementação como na resposta anterior)
    (tl, tr, br, bl) = pontos_pagina_ordenados_np
    dst_pts = np.array([[0, 0], [ROI_LARGURA_SAIDA_PARA_OCR - 1, 0], [ROI_LARGURA_SAIDA_PARA_OCR - 1, ROI_ALTURA_SAIDA_PARA_OCR - 1], [0, ROI_ALTURA_SAIDA_PARA_OCR - 1]], dtype="float32")
    matrix_perspectiva = cv2.getPerspectiveTransform(pontos_pagina_ordenados_np, dst_pts)
    roi_corrigida = cv2.warpPerspective(imagem_cv_original, matrix_perspectiva, (ROI_LARGURA_SAIDA_PARA_OCR, ROI_ALTURA_SAIDA_PARA_OCR))
    return roi_corrigida

def preprocessar_para_ocr(roi_corrigida_cv):
    # (Implementação como na resposta anterior, usando constantes globais para parâmetros)
    # ...
    try:
        if roi_corrigida_cv is None: print("Erro Pré-OCR: ROI corrigida Nula"); return None, None
        if len(roi_corrigida_cv.shape) == 3: gray = cv2.cvtColor(roi_corrigida_cv, cv2.COLOR_BGR2GRAY)
        else: gray = roi_corrigida_cv.copy()
        print(f"Pré-OCR: Dimensões da ROI cinza: {gray.shape}")
        processed_img = gray
        if APLICAR_MEDIAN_BLUR_OCR:
            processed_img = cv2.medianBlur(processed_img, MEDIAN_BLUR_KERNEL_SIZE_OCR)
            print(f"Pré-OCR: Aplicado Median Blur com kernel {MEDIAN_BLUR_KERNEL_SIZE_OCR}")
        if APLICAR_NITIDEZ_OCR:
            kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed_img = cv2.filter2D(processed_img, -1, kernel_sharpening)
            print("Pré-OCR: Aplicado filtro de nitidez.")
        print(f"Pré-OCR: Aplicando Adaptive Threshold: BlockSize={ADAPTIVE_THRESH_BLOCK_SIZE_OCR}, C={ADAPTIVE_THRESH_C_OCR}")
        binarized = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE_OCR, ADAPTIVE_THRESH_C_OCR)
        if INVERTER_BINARIZACAO_FINAL_OCR: print("Pré-OCR: Invertendo imagem binarizada FINAL."); binarized = cv2.bitwise_not(binarized)
        processed_img = binarized
        if APLICAR_MORFOLOGIA_OPEN_OCR:
            kernel_morph = np.ones(KERNEL_MORFOLOGIA_OCR_SIZE, np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel_morph, iterations=1)
            print(f"Pré-OCR: Aplicada morfologia OPEN com kernel {KERNEL_MORFOLOGIA_OCR_SIZE}")
        if APLICAR_MORFOLOGIA_CLOSE_OCR:
            kernel_morph = np.ones(KERNEL_MORFOLOGIA_OCR_SIZE, np.uint8)
            processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel_morph, iterations=1)
            print(f"Pré-OCR: Aplicada morfologia CLOSE com kernel {KERNEL_MORFOLOGIA_OCR_SIZE}")
        cv2.imwrite(ROI_PROCESSADA_PATH, processed_img)
        print(f"Pré-OCR: Imagem final para Tesseract salva em: {ROI_PROCESSADA_PATH} (VERIFIQUE ESTA IMAGEM!)")
        img_para_gui = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR) if len(processed_img.shape) == 2 else processed_img
        return ROI_PROCESSADA_PATH, img_para_gui
    except Exception as e: print(f"Erro em preprocessar_para_ocr: {e}"); traceback.print_exc(); return None, None


def executar_ocr_tesseract(caminho_imagem_para_ocr, idioma=TESSERACT_IDIOMA, psm=TESSERACT_PSM_MODE):
    # (Implementação como na resposta anterior)
    # ...
    try:
        comando_tesseract_lista_str = f'tesseract "{caminho_imagem_para_ocr}" stdout -l {idioma} --psm {psm}'
        print(f"Executando Tesseract: {comando_tesseract_lista_str}")
        process = subprocess.Popen(comando_tesseract_lista_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        stdout_data, stderr_data = process.communicate(timeout=45)
        if process.returncode != 0: print(f"Erro Tesseract ({process.returncode}):\n{stderr_data}"); return ""
        if stderr_data.strip(): print(f"Stderr Tesseract (avisos):\n{stderr_data.strip()}")
        linhas_texto = [linha.strip() for linha in stdout_data.split('\n') if linha.strip()]
        texto_limpo = "\n".join(linhas_texto)
        if not texto_limpo: print("Tesseract não retornou texto (saída limpa vazia).")
        else: print(f"Tesseract retornou {len(texto_limpo)} caracteres.")
        return texto_limpo
    except subprocess.TimeoutExpired: print("Tesseract timeout."); falar_texto_windows("O OCR demorou demais."); return ""
    except FileNotFoundError: print("Tesseract não encontrado."); falar_texto_windows("Programa OCR não encontrado."); return ""
    except Exception as e: print(f"Erro OCR: {e}"); traceback.print_exc(); falar_texto_windows("Erro no OCR."); return ""

# --- FUNÇÃO processar_texto_com_marcadores ATUALIZADA ---
def processar_texto_com_marcadores(texto_completo_ocr, app_ref):
    print("\n--- Processando Texto com Marcadores (Novos Marcadores) ---")
    linhas_do_texto = texto_completo_ocr.split('\n')
    # Usa o estado da aplicação para persistir se está lendo conteúdo
    lendo_conteudo_nesta_pagina = app_ref.deve_ler_conteudo_principal_doc 
    status_pagina_atual = "CONTINUAR_LEITURA_PAGINA" 
    
    print(f"Estado inicial de 'lendo_conteudo': {lendo_conteudo_nesta_pagina}")

    for linha_atual in linhas_do_texto:
        linha_limpa = linha_atual.strip()
        if not linha_limpa: continue
        
        print(f"Analisando linha: '{linha_limpa}'") 
        marcador_identificado_na_linha = None
        texto_complementar_do_marcador = ""
        
        for marcador_chave, frase_audio_associada in MARCADORES_AUDIO.items():
            # Usar startswith para os marcadores que têm conteúdo depois (TITULO:, INICIO:)
            # Usar igualdade exata para marcadores de comando (PULA_PAGINA, FINAL)
            # Ou, para maior robustez, verificar se a linha limpa *contém* o marcador
            # e se for o caso, tentar extrair o conteúdo.
            # Por simplicidade, vamos manter startswith para os com ':' e igualdade para os sem.

            if marcador_chave.endswith(":"):
                if linha_limpa.upper().startswith(marcador_chave.upper()): # Ignora case para o marcador
                    marcador_identificado_na_linha = marcador_chave
                    # Extrai o que vem depois do marcador (incluindo o ':')
                    try:
                        # Encontra o índice do final do marcador_chave na linha_limpa (case insensitive)
                        idx_marcador_fim = linha_limpa.upper().find(marcador_chave.upper()) + len(marcador_chave)
                        texto_complementar_do_marcador = linha_limpa[idx_marcador_fim:].strip()
                    except: # Fallback
                        partes_linha = linha_limpa.split(":", 1)
                        if len(partes_linha) > 1:
                            texto_complementar_do_marcador = partes_linha[1].strip()
            elif linha_limpa.upper() == marcador_chave.upper(): # Para PULA_PAGINA, FINAL (sem :)
                marcador_identificado_na_linha = marcador_chave

            if marcador_identificado_na_linha:
                print(f"  -> Marcador ENCONTRADO: {marcador_chave}")
                falar_texto_windows(frase_audio_associada)
                
                if marcador_chave == "TITULO:":
                    if texto_complementar_do_marcador: falar_texto_windows(texto_complementar_do_marcador)
                    lendo_conteudo_nesta_pagina = False 
                    print(f"    -> Lendo_conteudo definido para: {lendo_conteudo_nesta_pagina} (após Título)")
                
                elif marcador_chave == "INICIO:":
                    lendo_conteudo_nesta_pagina = True
                    print(f"    -> Lendo_conteudo definido para: {lendo_conteudo_nesta_pagina} (após Início)")
                    if texto_complementar_do_marcador: falar_texto_windows(texto_complementar_do_marcador)
                
                elif marcador_chave == "PULA_PAGINA":
                    status_pagina_atual = "PULAR_PAGINA"
                    app_ref.deve_ler_conteudo_principal_doc = lendo_conteudo_nesta_pagina
                    return status_pagina_atual
                
                elif marcador_chave == "FINAL":
                    status_pagina_atual = "FINAL_DOCUMENTO"
                    lendo_conteudo_nesta_pagina = False
                    app_ref.deve_ler_conteudo_principal_doc = lendo_conteudo_nesta_pagina
                    return status_pagina_atual
                break # Sai do loop de marcadores se um foi encontrado
        
        if not marcador_identificado_na_linha and lendo_conteudo_nesta_pagina:
            print(f"  -> Falando linha de conteúdo: '{linha_limpa}'")
            falar_texto_windows(linha_limpa)
        elif not marcador_identificado_na_linha and not lendo_conteudo_nesta_pagina:
            print(f"  -> Linha ignorada (não é marcador e não está lendo conteúdo): '{linha_limpa}'")

    app_ref.deve_ler_conteudo_principal_doc = lendo_conteudo_nesta_pagina
    print(f"Estado final de 'lendo_conteudo' para próxima página (se houver): {app_ref.deve_ler_conteudo_principal_doc}")
    print("--- Fim do Processamento de Marcadores ---")
    return status_pagina_atual

# --- Classe LeitorDocumentosApp ---
# (Definição da classe como na resposta anterior, garantindo que _logica_analise_pagina
#  chama a função global processar_texto_com_marcadores(texto_ocr_da_pagina, self)
#  e que os métodos auxiliares da GUI (atualizar_status_gui, etc.) estão presentes)
class LeitorDocumentosApp:
    # ... (COPIE A DEFINIÇÃO COMPLETA DA CLASSE LeitorDocumentosApp DA RESPOSTA ANTERIOR AQUI) ...
    # Verifique se todas as chamadas para processar_e_ler_conteudo_documento_gui
    # foram substituídas por processar_texto_com_marcadores(texto_ocr, self)
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Leitor de Documentos Otimizado")
        self.root.geometry("1100x900" if DEBUG_VISUAL or CAMINHO_IMAGEM_TESTE else "950x800")
        self.status_geral_leitura = "INICIO"
        self.deve_ler_conteudo_principal_doc = False
        self.imagem_carregada_cv = None 
        self.webcam_idx = 0; self.webcam_backend = None; self.webcam = None 
        self.latest_frame_cv = None; self.video_thread_running = False
        self.video_thread = None; self.frame_lock = threading.Lock()
        self.tts_worker_thread_ref = tts_worker_thread
        self._setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.ao_fechar_janela)
        if CAMINHO_IMAGEM_TESTE: self.carregar_e_exibir_imagem_estatica()
        else: self.iniciar_feed_video()

    def _setup_gui(self):
        self.frame_controles = ttk.LabelFrame(self.root, text="Controles", padding=10)
        self.frame_controles.pack(pady=5, padx=10, fill="x")
        self.btn_analisar_pagina = ttk.Button(self.frame_controles, text="Analisar Imagem/Página", command=self.disparar_analise_pagina)
        self.btn_analisar_pagina.pack(side=tk.LEFT, padx=5)
        if CAMINHO_IMAGEM_TESTE is None:
            self.btn_parar_video = ttk.Button(self.frame_controles, text="Parar Feed", command=self.parar_feed_video, state=tk.DISABLED)
            self.btn_parar_video.pack(side=tk.LEFT, padx=5)
            self.btn_iniciar_video = ttk.Button(self.frame_controles, text="Iniciar Feed", command=self.iniciar_feed_video)
            self.btn_iniciar_video.pack(side=tk.LEFT, padx=5)
        self.lbl_status_gui = ttk.Label(self.frame_controles, text="Status: Aguardando...")
        self.lbl_status_gui.pack(side=tk.LEFT, padx=10, fill="x", expand=True)
        self.frame_principal_imagens = ttk.Frame(self.root, padding=10)
        self.frame_principal_imagens.pack(pady=5, padx=10, fill="both", expand=True)
        label_img_principal = "Imagem Carregada (c/ Detecções)" if CAMINHO_IMAGEM_TESTE else "Feed da Webcam"
        self.frame_img_principal_gui = ttk.LabelFrame(self.frame_principal_imagens, text=label_img_principal, padding=5)
        self.frame_img_principal_gui.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_img_principal_display = ttk.Label(self.frame_img_principal_gui)
        self.lbl_img_principal_display.pack(fill="both", expand=True)
        self.frame_img_roi_gui = ttk.LabelFrame(self.frame_principal_imagens, text="ROI Processada p/ OCR", padding=5)
        self.frame_img_roi_gui.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.lbl_img_roi_processada = ttk.Label(self.frame_img_roi_gui)
        self.lbl_img_roi_processada.pack(fill="both", expand=True)
        self.frame_principal_imagens.columnconfigure(0, weight=1); self.frame_principal_imagens.columnconfigure(1, weight=1)
        self.frame_principal_imagens.rowconfigure(0, weight=1)
        if DEBUG_VISUAL:
            self.frame_debug_imagens = ttk.LabelFrame(self.root, text="Depuração Detecção Quadrados", padding=10)
            self.frame_debug_imagens.pack(pady=5, padx=10, fill="x")
            self.frame_img_thresh = ttk.LabelFrame(self.frame_debug_imagens, text="Binarizada (Detecção)", padding=5)
            self.frame_img_thresh.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            self.lbl_img_thresh_debug = ttk.Label(self.frame_img_thresh)
            self.lbl_img_thresh_debug.pack(fill="both", expand=True)
            self.frame_img_contornos = ttk.LabelFrame(self.frame_debug_imagens, text="Contornos (Detecção)", padding=5)
            self.frame_img_contornos.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
            self.lbl_img_contornos_debug = ttk.Label(self.frame_img_contornos)
            self.lbl_img_contornos_debug.pack(fill="both", expand=True)
            self.frame_debug_imagens.columnconfigure(0, weight=1); self.frame_debug_imagens.columnconfigure(1, weight=1)
        self.frame_texto_ocr = ttk.LabelFrame(self.root, text="Texto Reconhecido", padding=10)
        self.frame_texto_ocr.pack(pady=5, padx=10, fill="x")
        self.txt_ocr_display = scrolledtext.ScrolledText(self.frame_texto_ocr, wrap=tk.WORD, height=8, state=tk.DISABLED)
        self.txt_ocr_display.pack(fill="both", expand=True)

    def atualizar_status_gui(self, mensagem):
        if hasattr(self, 'lbl_status_gui') and self.lbl_status_gui.winfo_exists(): self.lbl_status_gui.config(text=f"Status: {mensagem}")
        else: print(f"STATUS_GUI: {mensagem}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def exibir_texto_ocr(self, texto):
        if hasattr(self, 'txt_ocr_display') and self.txt_ocr_display.winfo_exists():
            self.txt_ocr_display.config(state=tk.NORMAL); self.txt_ocr_display.delete(1.0, tk.END); self.txt_ocr_display.insert(tk.END, texto); self.txt_ocr_display.config(state=tk.DISABLED)
        else: print(f"TEXTO_OCR_GUI: {texto}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def exibir_imagem_em_label(self, img_cv, label_widget, largura_max, altura_max):
        if not (hasattr(label_widget, 'winfo_exists') and label_widget.winfo_exists()): return
        if img_cv is None: label_widget.config(image=''); label_widget.image = None; return
        img_cv_copy = img_cv.copy(); h, w = img_cv_copy.shape[:2]
        if w == 0 or h == 0: label_widget.config(image=''); label_widget.image = None; return
        escala = min(largura_max/w, altura_max/h) if w > 0 and h > 0 else 1
        nova_largura, nova_altura = int(w * escala), int(h * escala)
        if nova_largura <= 0 or nova_altura <= 0: label_widget.config(image=''); label_widget.image = None; return
        try:
            img_redimensionada = cv2.resize(img_cv_copy, (nova_largura, nova_altura)); img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb); img_tk = ImageTk.PhotoImage(image=img_pil)
            label_widget.config(image=img_tk); label_widget.image = img_tk
        except Exception as e: print(f"Erro exibir_imagem: {e}"); traceback.print_exc(); label_widget.config(image=''); label_widget.image = None;
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def carregar_e_exibir_imagem_estatica(self):
        self.atualizar_status_gui(f"Carregando: {os.path.basename(CAMINHO_IMAGEM_TESTE) if CAMINHO_IMAGEM_TESTE else 'N/A'}")
        if not CAMINHO_IMAGEM_TESTE: self.btn_analisar_pagina.config(state=tk.DISABLED); return
        try:
            if not os.path.exists(CAMINHO_IMAGEM_TESTE):
                msg = f"Erro: Arquivo não encontrado: {CAMINHO_IMAGEM_TESTE}"; print(msg); self.atualizar_status_gui(msg); falar_texto_windows(msg)
                if hasattr(self, 'btn_analisar_pagina'): self.btn_analisar_pagina.config(state=tk.DISABLED); return
            self.imagem_carregada_cv = cv2.imread(CAMINHO_IMAGEM_TESTE)
            if self.imagem_carregada_cv is None:
                msg = f"Erro: Não foi possível ler: {CAMINHO_IMAGEM_TESTE}"; print(msg); self.atualizar_status_gui(msg); falar_texto_windows(msg)
                if hasattr(self, 'btn_analisar_pagina'): self.btn_analisar_pagina.config(state=tk.DISABLED); return
            print(f"Imagem {CAMINHO_IMAGEM_TESTE} carregada."); 
            self.exibir_imagem_em_label(self.imagem_carregada_cv.copy(), self.lbl_img_principal_display, IMG_LARGURA_GUI_PRINCIPAL, IMG_ALTURA_GUI_PRINCIPAL)
            self.atualizar_status_gui("Imagem carregada. Clique 'Analisar'."); falar_texto_windows("Imagem carregada.")
            if hasattr(self, 'btn_analisar_pagina'): self.btn_analisar_pagina.config(state=tk.NORMAL)
        except Exception as e:
            msg = f"Exceção ao carregar imagem: {e}"; print(msg); traceback.print_exc(); self.atualizar_status_gui(msg); falar_texto_windows("Erro ao carregar imagem.")
            if hasattr(self, 'btn_analisar_pagina'): self.btn_analisar_pagina.config(state=tk.DISABLED)

    def iniciar_feed_video(self):
        if CAMINHO_IMAGEM_TESTE: print("Modo de imagem estática. Feed de vídeo não será iniciado."); return
        if self.video_thread_running: print("Feed já rodando."); return
        self.atualizar_status_gui("Iniciando feed...")
        try:
            cap_args = [self.webcam_idx]
            if self.webcam_backend is not None: cap_args.append(self.webcam_backend)
            self.webcam = cv2.VideoCapture(*cap_args)
            if not self.webcam.isOpened():
                print(f"Webcam {self.webcam_idx} (padrão/configurado) falhou. Tentando DSHOW..."); 
                self.webcam = cv2.VideoCapture(self.webcam_idx, cv2.CAP_DSHOW); self.webcam_backend = cv2.CAP_DSHOW
        except Exception as e: print(f"Exceção ao abrir webcam: {e}"); self.webcam = None
        if not self.webcam or not self.webcam.isOpened():
            self.atualizar_status_gui(f"Erro: Webcam {self.webcam_idx} não abriu."); falar_texto_windows(f"Webcam {self.webcam_idx} não iniciou."); self.webcam = None; 
            if hasattr(self,'btn_iniciar_video'): self.btn_iniciar_video.config(state=tk.NORMAL)
            if hasattr(self,'btn_parar_video'): self.btn_parar_video.config(state=tk.DISABLED)
            return
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_w = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH); actual_h = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam {self.webcam_idx} (backend: {self.webcam_backend}) aberta. Res: {actual_w}x{actual_h}")
        self.video_thread_running = True
        self.video_thread = threading.Thread(target=self._loop_feed_video, daemon=True); self.video_thread.start()
        if hasattr(self,'btn_parar_video'): self.btn_parar_video.config(state=tk.NORMAL)
        if hasattr(self,'btn_iniciar_video'): self.btn_iniciar_video.config(state=tk.DISABLED)
        self._atualizar_label_feed_video()

    def _loop_feed_video(self):
        print("Thread feed vídeo: Iniciada.")
        try:
            while self.video_thread_running:
                if self.webcam and self.webcam.isOpened():
                    ret, frame = self.webcam.read()
                    if ret and frame is not None:
                        with self.frame_lock: self.latest_frame_cv = frame.copy()
                    else: print("Feed: Falha ao ler frame."); time.sleep(0.1) 
                else: print("Feed: Webcam não aberta."); self.video_thread_running = False; break 
                time.sleep(1/30) 
        except Exception as e: print(f"Exceção no feed: {e}"); traceback.print_exc()
        finally:
            if self.webcam: self.webcam.release(); self.webcam = None
            self.video_thread_running = False; print("Thread feed vídeo: Finalizada.")
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, lambda: self.btn_iniciar_video.config(state=tk.NORMAL) if hasattr(self,'btn_iniciar_video') and self.btn_iniciar_video.winfo_exists() else None)
                self.root.after(0, lambda: self.btn_parar_video.config(state=tk.DISABLED) if hasattr(self,'btn_parar_video') and self.btn_parar_video.winfo_exists() else None)

    def _atualizar_label_feed_video(self):
        if not (hasattr(self, 'root') and self.root.winfo_exists()): return
        if not self.video_thread_running:
            if hasattr(self, 'lbl_img_principal_display') and self.lbl_img_principal_display.winfo_exists():
                 self.lbl_img_principal_display.config(image=''); self.lbl_img_principal_display.image = None
            return
        frame_para_exibir = None; 
        with self.frame_lock:
            if self.latest_frame_cv is not None: frame_para_exibir = self.latest_frame_cv.copy()
        if frame_para_exibir is not None and hasattr(self, 'lbl_img_principal_display') and self.lbl_img_principal_display.winfo_exists():
            self.exibir_imagem_em_label(frame_para_exibir, self.lbl_img_principal_display, IMG_LARGURA_GUI_PRINCIPAL, IMG_ALTURA_GUI_PRINCIPAL)
        if self.video_thread_running: self.root.after(33, self._atualizar_label_feed_video)

    def parar_feed_video(self):
        self.atualizar_status_gui("Parando feed...")
        self.video_thread_running = False
        if self.video_thread and self.video_thread.is_alive(): self.video_thread.join(timeout=1.0)
        if hasattr(self, 'lbl_img_principal_display') and self.lbl_img_principal_display.winfo_exists():
            self.lbl_img_principal_display.config(image=''); self.lbl_img_principal_display.image = None
        if hasattr(self,'btn_parar_video'): self.btn_parar_video.config(state=tk.DISABLED)
        if hasattr(self,'btn_iniciar_video'): self.btn_iniciar_video.config(state=tk.NORMAL)
        self.atualizar_status_gui("Feed parado.")

    def disparar_analise_pagina(self):
        if self.status_geral_leitura == "FINAL_DOCUMENTO":
            self.atualizar_status_gui("Documento finalizado."); falar_texto_windows("Documento já lido."); return
        frame_para_processar = None
        if CAMINHO_IMAGEM_TESTE:
            if self.imagem_carregada_cv is None:
                self.atualizar_status_gui("Nenhuma imagem carregada."); falar_texto_windows("Carregue uma imagem."); return
            frame_para_processar = self.imagem_carregada_cv.copy()
        else: 
            if not self.video_thread_running or self.latest_frame_cv is None:
                self.atualizar_status_gui("Feed não ativo."); falar_texto_windows("Inicie o feed."); return
            with self.frame_lock:
                if self.latest_frame_cv is not None: frame_para_processar = self.latest_frame_cv.copy()
        if frame_para_processar is None: self.atualizar_status_gui("Sem frame para análise."); return
        if hasattr(self,'lbl_img_roi_processada'): self.exibir_imagem_em_label(None, self.lbl_img_roi_processada, IMG_LARGURA_GUI_ROI_PROC, IMG_ALTURA_GUI_ROI_PROC)
        if DEBUG_VISUAL and hasattr(self,'lbl_img_thresh_debug'): self.exibir_imagem_em_label(None, self.lbl_img_thresh_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
        if DEBUG_VISUAL and hasattr(self,'lbl_img_contornos_debug'): self.exibir_imagem_em_label(None, self.lbl_img_contornos_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
        self.exibir_texto_ocr("Analisando imagem...")
        thread_analise = threading.Thread(target=self._logica_analise_pagina, args=(frame_para_processar,), daemon=True)
        thread_analise.start()

    def _logica_analise_pagina(self, frame_capturado_cv):
        btn_analise = getattr(self, 'btn_analisar_pagina', None)
        try:
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.DISABLED)
            self.atualizar_status_gui("Analisando delimitadores...")
            pontos_pagina, img_com_deteccoes_gui = detectar_delimitadores_pagina(frame_capturado_cv, self) 
 # Atualiza o painel principal com a imagem que teve as detecções (seja webcam ou estática)
            if CAMINHO_IMAGEM_TESTE and img_com_deteccoes_gui is not None and hasattr(self, 'lbl_img_principal_display'):
               self.exibir_imagem_em_label(img_com_deteccoes_gui, self.lbl_img_principal_display, IMG_LARGURA_GUI_PRINCIPAL, IMG_ALTURA_GUI_PRINCIPAL) # <--- CORRIGIDO AQUI
            if pontos_pagina is None:
                self.atualizar_status_gui(f"Não detectados 4 delimitadores. Ajuste params/imagem."); falar_texto_windows("Não encontrei os cantos da página.");
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            self.atualizar_status_gui("Corrigindo perspectiva...")
            roi_cv_retificada = corrigir_perspectiva(frame_capturado_cv, pontos_pagina)
            if roi_cv_retificada is None:
                self.atualizar_status_gui("Erro perspectiva."); falar_texto_windows("Erro ao ajustar imagem.");
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            self.atualizar_status_gui("Pré-processando para OCR...")
            caminho_roi_para_ocr, img_roi_proc_gui = preprocessar_para_ocr(roi_cv_retificada)
            if img_roi_proc_gui is not None and hasattr(self, 'lbl_img_roi_processada'):
                self.exibir_imagem_em_label(img_roi_proc_gui, self.lbl_img_roi_processada, IMG_LARGURA_GUI_ROI_PROC, IMG_ALTURA_GUI_ROI_PROC)
            if caminho_roi_para_ocr is None:
                self.atualizar_status_gui("Erro pré-processamento OCR."); falar_texto_windows("Erro ao preparar imagem.");
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            self.atualizar_status_gui("Realizando OCR...")
            texto_ocr_da_pagina = executar_ocr_tesseract(caminho_roi_para_ocr)
            self.exibir_texto_ocr(texto_ocr_da_pagina) 
            if not texto_ocr_da_pagina.strip() and self.status_geral_leitura != "FINAL_DOCUMENTO":
                self.atualizar_status_gui("OCR sem texto. Verifique img processada e params."); falar_texto_windows("Nenhum texto reconhecido.");
                self.status_geral_leitura = "ERRO_OCR" if CAMINHO_IMAGEM_TESTE else "PULAR_PAGINA"
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            self.atualizar_status_gui("OCR OK. Processando marcadores e narrando...")
            # Chama a função global para processar marcadores, passando a referência da app
            self.status_geral_leitura = processar_texto_com_marcadores(texto_ocr_da_pagina, self)
            if self.status_geral_leitura == "CONTINUAR_LEITURA_PAGINA":
                msg_fim = "Fim da análise da imagem." if CAMINHO_IMAGEM_TESTE else "Fim conteúdo página. Próxima?"
                self.atualizar_status_gui(msg_fim); falar_texto_windows(msg_fim) 
                if not CAMINHO_IMAGEM_TESTE: self.status_geral_leitura = "PULAR_PAGINA"
            if self.status_geral_leitura == "FINAL_DOCUMENTO": self.atualizar_status_gui("Documento Finalizado!")
            elif self.status_geral_leitura == "PULAR_PAGINA" and not CAMINHO_IMAGEM_TESTE: self.atualizar_status_gui("Pronto para próxima página.")
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL)
        except Exception as e:
            self.atualizar_status_gui(f"Erro análise: {str(e)[:50]}")
            print(f"Erro thread análise: {e}"); traceback.print_exc()
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL)

    def ao_fechar_janela(self):
        print("Fechando aplicação...")
        self.video_thread_running = False
        if self.video_thread and self.video_thread.is_alive():
            print("Aguardando thread vídeo..."); self.video_thread.join(timeout=1.0) 
        
        if tts_engine and hasattr(self, 'tts_worker_thread_ref') and self.tts_worker_thread_ref.is_alive():
             tts_queue.put(None); self.tts_worker_thread_ref.join(timeout=1.0)
        if tts_engine:
            try: tts_engine.stop()
            except RuntimeError as e: print(f"Ignorando erro TTS ao fechar: {e}")
            except Exception as e: print(f"Erro ao parar TTS: {e}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.destroy()
        print("Aplicação fechada.")

if __name__ == "__main__":
    if not os.path.exists(TEMP_DIR):
        try: os.makedirs(TEMP_DIR); print(f"Diretório temporário criado: {TEMP_DIR}")
        except OSError as e:
            print(f"Não foi possível criar {TEMP_DIR}: {e}. Usando diretório atual."); TEMP_DIR = os.getcwd()
            
    root = tk.Tk()
    app = LeitorDocumentosApp(root)
    if 'tts_worker_thread' in globals() and tts_worker_thread is not None and tts_worker_thread.is_alive():
        app.tts_worker_thread_ref = tts_worker_thread 
    root.mainloop()