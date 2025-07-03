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
import traceback # Para melhor depuração de exceções

# --- Configurações e Constantes ---
DEBUG_VISUAL = True # MUITO IMPORTANTE: Deixe como True para depurar a detecção de quadrados!
                   # Isso mostrará imagens intermediárias na GUI.

TEMP_DIR = os.environ.get('TEMP', os.path.join(os.getcwd(), 'temp_ocr')) # Usa subpasta local se TEMP não acessível
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR) # Garante que o diretório TEMP exista

ROI_PERSPECTIVA_PATH = os.path.join(TEMP_DIR, "leitor_gui_roi_perspectiva.png")
ROI_PROCESSADA_PATH = os.path.join(TEMP_DIR, "leitor_gui_roi_processada.png")

# Parâmetros para detecção dos quadrados (AJUSTE ESTES VALORES CONFORME NECESSÁRIO!)
MIN_QUADRADO_AREA = 300     # Área mínima em pixels do quadrado (ajuste olhando o console)
MAX_QUADRADO_AREA = 40000   # Área máxima em pixels
PROPORCAO_QUADRADO_MIN = 0.70 # Mais tolerante para formas não perfeitamente quadradas
PROPORCAO_QUADRADO_MAX = 1.30 # Mais tolerante
GAUSSIAN_BLUR_KERNEL_DETECCAO = (5, 5) 
ADAPTIVE_THRESH_BLOCK_SIZE = 25 # Ímpar. Experimente 11, 15, 21, 25, 31, 41... Afeta "Img Binarizada"
ADAPTIVE_THRESH_C = 4          # Experimente 3, 5, 7, 9, 11... Afeta "Img Binarizada"
APPROX_POLY_EPSILON = 0.035    # Para cv2.approxPolyDP (0.02 a 0.045 do perímetro)

# Dimensões para exibição na GUI e para processamento da ROI
IMG_LARGURA_GUI_FEED = 400 
IMG_ALTURA_GUI_FEED = 300
IMG_LARGURA_GUI_DEBUG = 320 
IMG_ALTURA_GUI_DEBUG = 240
IMG_LARGURA_GUI_ROI = 400
IMG_ALTURA_GUI_ROI = 300

ROI_LARGURA_SAIDA_OCR = 800 # Resolução da ROI que vai para o OCR
ROI_ALTURA_SAIDA_OCR = 1100

MARCADORES_AUDIO = {
    "#_TITULO_": "Título do Texto",
    "#_INICIO_": "Início do texto",
    "#_PULA_PAGINA_": "Fim da página, por favor, passe para a próxima",
    "#_FINAL_": "Fim do Texto"
}

try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 170)
    voices = tts_engine.getProperty('voices')
    for voice in voices:
        if "brazil" in voice.name.lower() or "portuguese" in voice.name.lower() or \
           "luciana" in voice.name.lower() or "maria" in voice.name.lower() or \
           (hasattr(voice, 'languages') and voice.languages and 'pt_BR' in voice.languages[0]):
            tts_engine.setProperty('voice', voice.id)
            print(f"Voz TTS definida para: {voice.name}")
            break
except Exception as e:
    print(f"Erro ao inicializar o motor TTS (pyttsx3): {e}")
    tts_engine = None

def falar_texto_windows(texto):
    if tts_engine:
        print(f"AUDIO: {texto}")
        try:
            def _speak():
                tts_engine.say(texto)
                tts_engine.runAndWait()
            threading.Thread(target=_speak, daemon=True).start()
        except Exception as e:
            print(f"Erro durante a fala (TTS): {e}")
    else:
        print(f"TTS indisponível. AUDIO (simulado): {texto}")

def order_points(pts_array_numpy):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts_array_numpy.sum(axis=1)
    rect[0] = pts_array_numpy[np.argmin(s)]
    rect[2] = pts_array_numpy[np.argmax(s)]
    diff = np.diff(pts_array_numpy, axis=1)
    rect[1] = pts_array_numpy[np.argmin(diff)]
    rect[3] = pts_array_numpy[np.argmax(diff)]
    return rect

def detectar_quadrados_e_obter_pontos_pagina(imagem_cv_original, app_ref_para_debug_gui):
    imagem_para_detectar = imagem_cv_original.copy()
    gray = cv2.cvtColor(imagem_para_detectar, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_DETECCAO, 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 
                                   ADAPTIVE_THRESH_BLOCK_SIZE, 
                                   ADAPTIVE_THRESH_C)
    
    if DEBUG_VISUAL and app_ref_para_debug_gui and hasattr(app_ref_para_debug_gui, 'lbl_img_thresh_debug'):
        app_ref_para_debug_gui.exibir_imagem_em_label(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), 
                                                 app_ref_para_debug_gui.lbl_img_thresh_debug, 
                                                 IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)

    # --- Opcional: Operações Morfológicas para limpar a imagem 'thresh' ---
    # Descomente e teste se a imagem 'thresh' tiver muito ruído ou buracos nos quadrados
    # kernel_morph = np.ones((3,3),np.uint8) # Kernel 3x3. Pode tentar (5,5)
    # thresh_processada = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=1)
    # thresh_processada = cv2.morphologyEx(thresh_processada, cv2.MORPH_CLOSE, kernel_morph, iterations=1)
    # if DEBUG_VISUAL and app_ref_para_debug_gui and hasattr(app_ref_para_debug_gui, 'lbl_img_contornos_debug'): # Ou um novo label
    #     app_ref_para_debug_gui.exibir_imagem_em_label(cv2.cvtColor(thresh_processada, cv2.COLOR_GRAY2BGR), 
    #                                              app_ref_para_debug_gui.lbl_img_contornos_debug, # Exemplo, idealmente outro label
    #                                              IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
    # contornos, _ = cv2.findContours(thresh_processada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se não usar operações morfológicas:
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    quadrados_candidatos = []
    img_com_contornos_debug = imagem_cv_original.copy()

    print("\n--- Depurando Detecção de Quadrados ---")
    print(f"Total de contornos encontrados inicialmente: {len(contornos)}")
    print(f"Usando: MIN_AREA={MIN_QUADRADO_AREA}, MAX_AREA={MAX_QUADRADO_AREA}, EPSILON_MULT={APPROX_POLY_EPSILON}, ASPECT_MIN/MAX={PROPORCAO_QUADRADO_MIN}/{PROPORCAO_QUADRADO_MAX}")

    for i, cnt in enumerate(contornos):
        area = cv2.contourArea(cnt)
        # Desenha todos os contornos iniciais (antes do filtro de área) em uma cor clara para referência
        # cv2.drawContours(img_com_contornos_debug, [cnt], -1, (200, 200, 200), 1) 

        if MIN_QUADRADO_AREA < area < MAX_QUADRADO_AREA:
            perimetro = cv2.arcLength(cnt, True)
            aprox_poligono = cv2.approxPolyDP(cnt, APPROX_POLY_EPSILON * perimetro, True)
            cv2.drawContours(img_com_contornos_debug, [aprox_poligono], -1, (255, 100, 0), 2) # Candidatos por área em azul

            if len(aprox_poligono) == 4 and cv2.isContourConvex(aprox_poligono):
                (x, y, w, h) = cv2.boundingRect(aprox_poligono)
                aspect_ratio = w / float(h) if h != 0 else 0
                print(f"  Candidato {i} (pós-área, 4 vértices): Área={area:.0f}, Vértices={len(aprox_poligono)}, Proporção={aspect_ratio:.2f}")

                if PROPORCAO_QUADRADO_MIN <= aspect_ratio <= PROPORCAO_QUADRADO_MAX:
                    M = cv2.moments(aprox_poligono)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        quadrados_candidatos.append({'centroid': np.array([cX, cY]), 'contour': aprox_poligono})
                        cv2.drawContours(img_com_contornos_debug, [aprox_poligono], -1, (0, 255, 0), 3) # Aceitos em verde
                        cv2.circle(img_com_contornos_debug, (cX, cY), 7, (0, 0, 255), -1) # Centroide em vermelho
                else: print(f"    -> Rejeitado por proporção ({aspect_ratio:.2f})")
            else: print(f"    -> Rejeitado: Vértices={len(aprox_poligono)}, Convexo={cv2.isContourConvex(aprox_poligono)}")
        # else: print(f"  Contorno {i}: Área={area:.0f} -> Rejeitado por filtro de área inicial")

    if DEBUG_VISUAL and app_ref_para_debug_gui and hasattr(app_ref_para_debug_gui, 'lbl_img_contornos_debug'):
        app_ref_para_debug_gui.exibir_imagem_em_label(img_com_contornos_debug, app_ref_para_debug_gui.lbl_img_contornos_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)

    if len(quadrados_candidatos) == 4:
        quadrados_centroids = [qc['centroid'] for qc in quadrados_candidatos]
        pontos_np = np.array(quadrados_centroids)
        pontos_ordenados = order_points(pontos_np)
        
        img_gui_com_deteccoes_finais = imagem_cv_original.copy()
        for qc in quadrados_candidatos:
             cv2.drawContours(img_gui_com_deteccoes_finais, [qc['contour']], -1, (0, 255, 0), 2)
        cores_pontos_ordenados = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        nomes_pontos_ordenados = ["SE", "SD", "ID", "IE"]
        for idx, pt_ord in enumerate(pontos_ordenados): # Renomeado 'i' para 'idx' e 'pt' para 'pt_ord'
            cv2.circle(img_gui_com_deteccoes_finais, tuple(pt_ord.astype(int)), 10, cores_pontos_ordenados[idx], -1)
            cv2.putText(img_gui_com_deteccoes_finais, nomes_pontos_ordenados[idx], tuple((pt_ord - [15,15]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        print("--- 4 quadrados delimitadores encontrados e ordenados com sucesso! ---")
        return pontos_ordenados, img_gui_com_deteccoes_finais
    else:
        msg = f"Detectados {len(quadrados_candidatos)} quadrados válidos. Esperava 4."
        print(f"--- {msg} ---")
        return None, img_com_contornos_debug

def aplicar_transformacao_perspectiva(imagem_cv_original, pontos_pagina_ordenados_np):
    (tl, tr, br, bl) = pontos_pagina_ordenados_np
    dst_pts = np.array([
        [0, 0], [ROI_LARGURA_SAIDA_OCR - 1, 0],
        [ROI_LARGURA_SAIDA_OCR - 1, ROI_ALTURA_SAIDA_OCR - 1], [0, ROI_ALTURA_SAIDA_OCR - 1]
    ], dtype="float32")
    matrix_perspectiva = cv2.getPerspectiveTransform(pontos_pagina_ordenados_np, dst_pts)
    imagem_retificada_roi = cv2.warpPerspective(imagem_cv_original, matrix_perspectiva, (ROI_LARGURA_SAIDA_OCR, ROI_ALTURA_SAIDA_OCR))
    return imagem_retificada_roi

def pre_processar_roi_para_ocr(imagem_roi_cv):
    try:
        if imagem_roi_cv is None: return None, None
        if len(imagem_roi_cv.shape) == 3: gray_roi = cv2.cvtColor(imagem_roi_cv, cv2.COLOR_BGR2GRAY)
        else: gray_roi = imagem_roi_cv.copy()
        
        # Para OCR, frequentemente uma leve nitidez ou não aplicar blur aqui pode ser melhor
        # gray_roi_processed = cv2.GaussianBlur(gray_roi, (3,3), 0) # Kernel pequeno, se necessário
        gray_roi_processed = gray_roi

        ret_otsu, binarized_roi = cv2.threshold(gray_roi_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Limiar de Otsu determinado para OCR: {ret_otsu}")
        
        # Tesseract geralmente prefere texto PRETO em fundo BRANCO.
        # Se Otsu resultou em texto branco em fundo preto, inverta.
        # Verifique a média de pixels: se > 127, o fundo é provavelmente branco.
        if np.mean(binarized_roi) < 100: # Heurística: se a imagem for predominantemente escura (texto branco)
           print("Invertendo ROI binarizada para OCR (texto preto em fundo branco)")
           binarized_roi = cv2.bitwise_not(binarized_roi)
        
        processed_roi_final = binarized_roi
        cv2.imwrite(ROI_PROCESSADA_PATH, processed_roi_final)
        print(f"ROI pré-processada para OCR salva em: {ROI_PROCESSADA_PATH}")
        
        img_para_gui = cv2.cvtColor(processed_roi_final, cv2.COLOR_GRAY2BGR) if len(processed_roi_final.shape) == 2 else processed_roi_final
        return ROI_PROCESSADA_PATH, img_para_gui
    except Exception as e:
        print(f"Erro no pré-processamento para OCR: {e}"); traceback.print_exc(); return None, None

def extrair_texto_da_imagem_com_ocr(caminho_imagem_processada, idioma_ocr='por', psm_mode='6'):
    try:
        # Aspas em torno do caminho são importantes se ele contiver espaços
        comando_tesseract_lista_str = f'tesseract "{caminho_imagem_processada}" stdout -l {idioma_ocr} --psm {psm_mode}'
        
        print(f"Executando Tesseract: {comando_tesseract_lista_str}")
        process = subprocess.Popen(comando_tesseract_lista_str,
                                   shell=True, # Necessário no Windows para interpretar o comando como string com aspas
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   encoding='utf-8', 
                                   errors='ignore')
        stdout_data, stderr_data = process.communicate(timeout=30)

        if process.returncode != 0: 
            print(f"Erro Tesseract (código {process.returncode}): {stderr_data}")
            return ""
        if stderr_data: print(f"Stderr Tesseract (pode conter avisos): {stderr_data.strip()}")
        
        linhas_texto = [linha.strip() for linha in stdout_data.split('\n') if linha.strip()]
        texto_limpo = "\n".join(linhas_texto)
        if not texto_limpo: print("OCR não retornou texto.")
        return texto_limpo
    except subprocess.TimeoutExpired: print("Tesseract timeout."); falar_texto_windows("O reconhecimento de texto demorou demais."); return ""
    except FileNotFoundError: print("Tesseract não encontrado."); falar_texto_windows("O programa de reconhecimento de texto não foi encontrado."); return ""
    except Exception as e: print(f"Erro OCR: {e}"); traceback.print_exc(); falar_texto_windows("Erro no OCR."); return ""

class LeitorDocumentosApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Leitor de Documentos Inteligente")
        self.root.geometry("1000x850" if DEBUG_VISUAL else "900x750") # Maior se debug visual

        self.status_geral_leitura = "INICIO"
        self.deve_ler_conteudo_principal_doc = False
        self.webcam_idx = 0
        self.webcam_backend = None 
        
        self.webcam = None 
        self.latest_frame_cv = None 
        self.video_thread_running = False
        self.video_thread = None
        self.frame_lock = threading.Lock()

        # --- Frame de Controles ---
        self.frame_controles = ttk.LabelFrame(self.root, text="Controles", padding=10)
        self.frame_controles.pack(pady=5, padx=10, fill="x")

        self.btn_iniciar_analise = ttk.Button(self.frame_controles, text="Analisar Página Atual", command=self.disparar_analise_pagina)
        self.btn_iniciar_analise.pack(side=tk.LEFT, padx=5)
        
        self.btn_parar_video = ttk.Button(self.frame_controles, text="Parar Feed", command=self.parar_feed_video)
        self.btn_parar_video.pack(side=tk.LEFT, padx=5); self.btn_parar_video.config(state=tk.DISABLED)

        self.btn_iniciar_video = ttk.Button(self.frame_controles, text="Iniciar Feed", command=self.iniciar_feed_video)
        self.btn_iniciar_video.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status_gui = ttk.Label(self.frame_controles, text="Status: Aguardando...")
        self.lbl_status_gui.pack(side=tk.LEFT, padx=10, fill="x", expand=True)

        # --- Frame Principal para Imagens (Feed e ROI) ---
        self.frame_principal_imagens = ttk.Frame(self.root, padding=10)
        self.frame_principal_imagens.pack(pady=5, padx=10, fill="both", expand=True)
        
        self.frame_img_feed = ttk.LabelFrame(self.frame_principal_imagens, text="Feed da Webcam", padding=5)
        self.frame_img_feed.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_img_feed_webcam = ttk.Label(self.frame_img_feed)
        self.lbl_img_feed_webcam.pack(fill="both", expand=True)

        self.frame_img_roi_gui = ttk.LabelFrame(self.frame_principal_imagens, text="Página Corrigida (ROI OCR)", padding=5)
        self.frame_img_roi_gui.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.lbl_img_roi_processada = ttk.Label(self.frame_img_roi_gui)
        self.lbl_img_roi_processada.pack(fill="both", expand=True)

        self.frame_principal_imagens.columnconfigure(0, weight=1); self.frame_principal_imagens.columnconfigure(1, weight=1)
        self.frame_principal_imagens.rowconfigure(0, weight=1)

        # --- Frame para Imagens de Depuração ---
        if DEBUG_VISUAL:
            self.frame_debug_imagens = ttk.LabelFrame(self.root, text="Depuração da Detecção", padding=10)
            self.frame_debug_imagens.pack(pady=5, padx=10, fill="x")

            self.frame_img_thresh = ttk.LabelFrame(self.frame_debug_imagens, text="Img Binarizada (Thresh Detect)", padding=5)
            self.frame_img_thresh.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
            self.lbl_img_thresh_debug = ttk.Label(self.frame_img_thresh) # Label para Thresh da Detecção
            self.lbl_img_thresh_debug.pack(fill="both", expand=True)

            self.frame_img_contornos = ttk.LabelFrame(self.frame_debug_imagens, text="Contornos Detectados (Debug)", padding=5)
            self.frame_img_contornos.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
            self.lbl_img_contornos_debug = ttk.Label(self.frame_img_contornos) # Label para Contornos da Detecção
            self.lbl_img_contornos_debug.pack(fill="both", expand=True)
            
            self.frame_debug_imagens.columnconfigure(0, weight=1); self.frame_debug_imagens.columnconfigure(1, weight=1)

        self.frame_texto_ocr = ttk.LabelFrame(self.root, text="Texto Reconhecido (OCR)", padding=10)
        self.frame_texto_ocr.pack(pady=5, padx=10, fill="x")
        self.txt_ocr_display = scrolledtext.ScrolledText(self.frame_texto_ocr, wrap=tk.WORD, height=6, state=tk.DISABLED)
        self.txt_ocr_display.pack(fill="both", expand=True)
        
        self.root.protocol("WM_DELETE_WINDOW", self.ao_fechar_janela)
        self.iniciar_feed_video()

    def atualizar_status_gui(self, mensagem):
        if hasattr(self, 'lbl_status_gui') and self.lbl_status_gui.winfo_exists():
            self.lbl_status_gui.config(text=f"Status: {mensagem}")
        else: print(f"STATUS (lbl_status_gui): {mensagem}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def exibir_texto_ocr(self, texto):
        if hasattr(self, 'txt_ocr_display') and self.txt_ocr_display.winfo_exists():
            self.txt_ocr_display.config(state=tk.NORMAL)
            self.txt_ocr_display.delete(1.0, tk.END)
            self.txt_ocr_display.insert(tk.END, texto)
            self.txt_ocr_display.config(state=tk.DISABLED)
        else: print(f"TEXTO OCR (txt_ocr_display): {texto}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def exibir_imagem_em_label(self, img_cv, label_widget, largura_max, altura_max):
        if not (hasattr(label_widget, 'winfo_exists') and label_widget.winfo_exists()): return
        if img_cv is None:
            label_widget.config(image=''); label_widget.image = None; return
        h, w = img_cv.shape[:2]
        if w == 0 or h == 0: label_widget.config(image=''); label_widget.image = None; return
        escala = min(largura_max/w, altura_max/h) if w > 0 and h > 0 else 1
        nova_largura, nova_altura = int(w * escala), int(h * escala)
        if nova_largura <= 0 or nova_altura <= 0: label_widget.config(image=''); label_widget.image = None; return
        
        try:
            img_redimensionada = cv2.resize(img_cv, (nova_largura, nova_altura))
            img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            label_widget.config(image=img_tk); label_widget.image = img_tk
        except Exception as e:
            print(f"Erro ao exibir imagem no label: {e}")
            label_widget.config(image=''); label_widget.image = None;
        
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.update_idletasks()

    def iniciar_feed_video(self):
        if self.video_thread_running: print("Feed já rodando."); return
        self.atualizar_status_gui("Iniciando feed...")
        
        # Força o uso de CAP_DSHOW se None falhar repetidamente ou causar problemas
        # Tente None primeiro, se der erro consistente, tente cv2.CAP_DSHOW
        try:
            if self.webcam_backend is None: 
                self.webcam = cv2.VideoCapture(self.webcam_idx) 
            else: 
                self.webcam = cv2.VideoCapture(self.webcam_idx, self.webcam_backend)
            
            if not self.webcam.isOpened(): # Tenta DSHOW como fallback
                print(f"Webcam {self.webcam_idx} com backend padrão/configurado falhou. Tentando DSHOW...")
                self.webcam = cv2.VideoCapture(self.webcam_idx, cv2.CAP_DSHOW)
                self.webcam_backend = cv2.CAP_DSHOW # Atualiza para refletir o backend usado
        except Exception as e:
            print(f"Exceção ao abrir webcam: {e}")
            self.webcam = None

        if not self.webcam or not self.webcam.isOpened():
            self.atualizar_status_gui(f"Erro: Webcam {self.webcam_idx} não abriu com nenhum backend tentado.")
            falar_texto_windows(f"Webcam {self.webcam_idx} não iniciou."); self.webcam = None; return

        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_w = self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Webcam {self.webcam_idx} (backend: {self.webcam_backend}) aberta. Resolução solicitada 640x480, obtida {actual_w}x{actual_h}")

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
                    else: 
                        print("Feed: Falha ao ler frame."); 
                        # Considerar parar o loop se falhas consecutivas ocorrerem
                        time.sleep(0.1) 
                else: 
                    print("Feed: Webcam não aberta ou tornou-se None."); 
                    self.video_thread_running = False; break 
                time.sleep(1/30) 
        except Exception as e: print(f"Exceção no feed: {e}"); traceback.print_exc()
        finally:
            if self.webcam: self.webcam.release(); self.webcam = None
            self.video_thread_running = False; print("Thread feed vídeo: Finalizada.")
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(0, lambda: self.btn_iniciar_video.config(state=tk.NORMAL) if hasattr(self,'btn_iniciar_video') and self.btn_iniciar_video.winfo_exists() else None)
                self.root.after(0, lambda: self.btn_parar_video.config(state=tk.DISABLED) if hasattr(self,'btn_parar_video') and self.btn_parar_video.winfo_exists() else None)

    def _atualizar_label_feed_video(self):
        if not (hasattr(self, 'root') and self.root.winfo_exists()): return # Root foi destruído
        if not self.video_thread_running:
            if hasattr(self, 'lbl_img_feed_webcam') and self.lbl_img_feed_webcam.winfo_exists():
                 self.lbl_img_feed_webcam.config(image=''); self.lbl_img_feed_webcam.image = None
            return
        frame_para_exibir = None
        with self.frame_lock:
            if self.latest_frame_cv is not None: frame_para_exibir = self.latest_frame_cv.copy()
        if frame_para_exibir is not None and hasattr(self, 'lbl_img_feed_webcam') and self.lbl_img_feed_webcam.winfo_exists():
            self.exibir_imagem_em_label(frame_para_exibir, self.lbl_img_feed_webcam, IMG_LARGURA_GUI_FEED, IMG_ALTURA_GUI_FEED)
        if self.video_thread_running: # Continua chamando 'after' apenas se o vídeo estiver rodando
            self.root.after(33, self._atualizar_label_feed_video)

    def parar_feed_video(self):
        self.atualizar_status_gui("Parando feed...")
        self.video_thread_running = False
        if self.video_thread and self.video_thread.is_alive(): self.video_thread.join(timeout=1.0)
        if hasattr(self, 'lbl_img_feed_webcam') and self.lbl_img_feed_webcam.winfo_exists():
            self.lbl_img_feed_webcam.config(image=''); self.lbl_img_feed_webcam.image = None
        if hasattr(self,'btn_parar_video'): self.btn_parar_video.config(state=tk.DISABLED)
        if hasattr(self,'btn_iniciar_video'): self.btn_iniciar_video.config(state=tk.NORMAL)
        self.atualizar_status_gui("Feed parado.")

    def disparar_analise_pagina(self):
        if self.status_geral_leitura == "FINAL_DOCUMENTO":
            self.atualizar_status_gui("Documento finalizado."); falar_texto_windows("Documento já lido."); return
        if not self.video_thread_running or self.latest_frame_cv is None:
            self.atualizar_status_gui("Feed não ativo."); falar_texto_windows("Inicie o feed."); return
        frame_para_processar = None
        with self.frame_lock:
            if self.latest_frame_cv is not None: frame_para_processar = self.latest_frame_cv.copy()
        if frame_para_processar is None: self.atualizar_status_gui("Sem frame para análise."); return
        
        if hasattr(self,'lbl_img_roi_processada'): self.exibir_imagem_em_label(None, self.lbl_img_roi_processada, IMG_LARGURA_GUI_ROI, IMG_ALTURA_GUI_ROI)
        if DEBUG_VISUAL and hasattr(self,'lbl_img_thresh_debug'): self.exibir_imagem_em_label(None, self.lbl_img_thresh_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
        if DEBUG_VISUAL and hasattr(self,'lbl_img_contornos_debug'): self.exibir_imagem_em_label(None, self.lbl_img_contornos_debug, IMG_LARGURA_GUI_DEBUG, IMG_ALTURA_GUI_DEBUG)
        
        self.exibir_texto_ocr("Analisando página...")
        thread_analise = threading.Thread(target=self._logica_analise_pagina, args=(frame_para_processar,), daemon=True)
        thread_analise.start()

    def _logica_analise_pagina(self, frame_capturado_cv):
        try:
            btn_analise = getattr(self, 'btn_iniciar_analise', getattr(self, 'btn_iniciar', None))
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.DISABLED)

            self.atualizar_status_gui("Analisando imagem...")
            pontos_pagina, img_com_deteccoes_snapshot = detectar_quadrados_e_obter_pontos_pagina(frame_capturado_cv, self) 
            
            # Opcional: Atualizar o feed principal com a imagem que teve as detecções para feedback visual imediato
            # if img_com_deteccoes_snapshot is not None and hasattr(self, 'lbl_img_feed_webcam'):
            #    self.exibir_imagem_em_label(img_com_deteccoes_snapshot, self.lbl_img_feed_webcam, IMG_LARGURA_GUI_FEED, IMG_ALTURA_GUI_FEED)

            if pontos_pagina is None:
                self.atualizar_status_gui(f"Não detectados 4 delimitadores. Ajuste a página e tente.")
                falar_texto_windows("Não encontrei os cantos da página. Por favor, ajuste.")
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL)
                return
            
            self.atualizar_status_gui("Delimitadores OK. Corrigindo perspectiva...")
            roi_cv_retificada = aplicar_transformacao_perspectiva(frame_capturado_cv, pontos_pagina)
            if roi_cv_retificada is None:
                self.atualizar_status_gui("Erro perspectiva."); falar_texto_windows("Erro ao ajustar imagem.");
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            
            self.atualizar_status_gui("Perspectiva OK. Pré-processando OCR...")
            caminho_roi_para_ocr, img_roi_proc_gui = pre_processar_roi_para_ocr(roi_cv_retificada)
            
            if img_roi_proc_gui is not None and hasattr(self, 'lbl_img_roi_processada'):
                self.exibir_imagem_em_label(img_roi_proc_gui, self.lbl_img_roi_processada, IMG_LARGURA_GUI_ROI, IMG_ALTURA_GUI_ROI)
            
            if caminho_roi_para_ocr is None:
                self.atualizar_status_gui("Erro pré-processamento OCR."); falar_texto_windows("Erro ao preparar imagem.");
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return
            
            self.atualizar_status_gui("Pré-processamento OK. Realizando OCR...")
            texto_ocr_da_pagina = extrair_texto_da_imagem_com_ocr(caminho_roi_para_ocr)
            self.exibir_texto_ocr(texto_ocr_da_pagina) 

            if not texto_ocr_da_pagina.strip() and self.status_geral_leitura != "FINAL_DOCUMENTO":
                self.atualizar_status_gui("OCR sem texto. Verifique."); falar_texto_windows("Nenhum texto reconhecido.");
                self.status_geral_leitura = "PULAR_PAGINA" 
                if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL); return

            self.atualizar_status_gui("OCR OK. Narrando...")
            status_processamento_pagina, self.deve_ler_conteudo_principal_doc = \
                self.processar_e_ler_conteudo_documento_gui(texto_ocr_da_pagina, self.deve_ler_conteudo_principal_doc)
            
            self.status_geral_leitura = status_processamento_pagina
            
            if self.status_geral_leitura == "CONTINUAR_LEITURA_PAGINA":
                self.atualizar_status_gui("Fim conteúdo página. Próxima?"); falar_texto_windows("Fim do conteúdo desta página.") 
                self.status_geral_leitura = "PULAR_PAGINA"
            
            if self.status_geral_leitura == "FINAL_DOCUMENTO": self.atualizar_status_gui("Documento Finalizado!")
            elif self.status_geral_leitura == "PULAR_PAGINA": self.atualizar_status_gui("Pronto para próxima página.")
            
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL)

        except Exception as e:
            self.atualizar_status_gui(f"Erro análise: {str(e)[:50]}")
            print(f"Erro thread análise: {e}"); traceback.print_exc()
            btn_analise = getattr(self, 'btn_iniciar_analise', getattr(self, 'btn_iniciar', None))
            if btn_analise and btn_analise.winfo_exists(): btn_analise.config(state=tk.NORMAL)

    def processar_e_ler_conteudo_documento_gui(self, texto_ocr_completo, estado_leitura_conteudo_anterior):
        linhas_do_texto = texto_ocr_completo.split('\n')
        lendo_conteudo_nesta_pagina = estado_leitura_conteudo_anterior
        status_pagina_atual = "CONTINUAR_LEITURA_PAGINA" 
        for linha_atual in linhas_do_texto:
            linha_limpa = linha_atual.strip()
            if not linha_limpa: continue
            marcador_identificado_na_linha = None
            texto_complementar_do_marcador = ""
            for marcador_chave, frase_audio_associada in MARCADORES_AUDIO.items():
                if linha_limpa.startswith(marcador_chave):
                    marcador_identificado_na_linha = marcador_chave
                    falar_texto_windows(frase_audio_associada)
                    partes_linha = linha_limpa.split(marcador_chave, 1)
                    if len(partes_linha) > 1: texto_complementar_do_marcador = partes_linha[1].strip()
                    if marcador_identificado_na_linha == "#_TITULO_":
                        if texto_complementar_do_marcador: falar_texto_windows(texto_complementar_do_marcador)
                        lendo_conteudo_nesta_pagina = False 
                    elif marcador_identificado_na_linha == "#_INICIO_":
                        lendo_conteudo_nesta_pagina = True
                        if texto_complementar_do_marcador: falar_texto_windows(texto_complementar_do_marcador)
                    elif marcador_identificado_na_linha == "#_PULA_PAGINA_":
                        status_pagina_atual = "PULAR_PAGINA"
                        return status_pagina_atual, lendo_conteudo_nesta_pagina
                    elif marcador_identificado_na_linha == "#_FINAL_":
                        status_pagina_atual = "FINAL_DOCUMENTO"
                        lendo_conteudo_nesta_pagina = False
                        return status_pagina_atual, lendo_conteudo_nesta_pagina
                    break 
            if not marcador_identificado_na_linha and lendo_conteudo_nesta_pagina:
                falar_texto_windows(linha_limpa)
        return status_pagina_atual, lendo_conteudo_nesta_pagina

    def ao_fechar_janela(self):
        print("Fechando aplicação...")
        self.video_thread_running = False
        if self.video_thread and self.video_thread.is_alive():
            print("Aguardando thread vídeo..."); self.video_thread.join(timeout=1.5) 
        if tts_engine:
            try: tts_engine.stop()
            except Exception as e: print(f"Erro ao parar TTS: {e}")
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.destroy()
        print("Aplicação fechada.")

if __name__ == "__main__":
    # Adicionado para garantir que a pasta TEMP_DIR exista ou possa ser criada
    if not os.path.exists(TEMP_DIR):
        try:
            os.makedirs(TEMP_DIR)
            print(f"Diretório temporário criado em: {TEMP_DIR}")
        except OSError as e:
            print(f"Não foi possível criar o diretório temporário {TEMP_DIR}: {e}")
            print("Por favor, crie manualmente ou defina TEMP_DIR para uma pasta com permissão de escrita.")
            # Considerar sair do script se TEMP_DIR não for utilizável
            # exit() 
            
    root = tk.Tk()
    app = LeitorDocumentosApp(root)
    root.mainloop()