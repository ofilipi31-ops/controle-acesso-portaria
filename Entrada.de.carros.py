import streamlit as st
import cv2
import sqlite3
import numpy as np
import time
import pickle
import easyocr
import re
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(page_title="Portaria Inteligente", layout="wide")

DB_NAME = "access_control.db"

# Inicialização do Banco de Dados SQLite (Exatamente as mesmas tabelas e dados de exemplo)
def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE NOT NULL,
            owner TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registered_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            face_data BLOB NOT NULL
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM vehicles')
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ('ABC1D23', 'Carlos Eduardo (Morador)', 'AUTHORIZED'),
            ('XYZ9W87', 'Visitante Proibido', 'BLOCKED'),
            ('BRA2E19', 'Ana Paula (Funcionária)', 'AUTHORIZED')
        ]
        cursor.executemany('INSERT INTO vehicles (plate, owner, status) VALUES (?, ?, ?)', sample_data)
        conn.commit()
    conn.close()

init_database()

# Carregamento do modelo EasyOCR (com cache para otimizar performance no Streamlit)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['pt', 'en'], gpu=False)

ocr_reader = load_ocr_reader()

# Funções de Lógica e Processamento (Exatamente as mesmas do original)[cite: 3]
def check_vehicle_database(plate):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT owner, status FROM vehicles WHERE plate = ?', (plate.upper(),))
    result = cursor.fetchone()
    conn.close()
    return result

def extract_face_features(face_roi):
    """ Extrai vetor de textura por matriz de blocos 4x4 pré-processados """[cite: 3]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    
    # Filtro CLAHE para normalizar sombra/luz[cite: 3]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Divisão em blocos 4x4 (16 sub-regiões)[cite: 3]
    grid_hists = []
    h, w = gray.shape
    step_h, step_w = h // 4, w // 4
    for row in range(4):
        for col in range(4):
            block = gray[row*step_h:(row+1)*step_h, col*step_w:(col+1)*step_w]
            hist = cv2.calcHist([block], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            grid_hists.extend(hist.flatten())
    return np.array(grid_hists, dtype=np.float32)

def check_face_database(face_roi):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT name, face_data FROM registered_faces')
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "Desconhecido", False
    current_features = extract_face_features(face_roi)
    best_score = -1.0
    best_name = "Desconhecido"
    for name, blob in rows:
        saved_features = pickle.loads(blob)
        if isinstance(saved_features, np.ndarray) and len(saved_features) == len(current_features):
            similarity = cv2.compareHist(
                current_features.reshape(-1, 1), 
                saved_features.reshape(-1, 1), 
                cv2.HISTCMP_CORREL
            )
        else:
            similarity = 0.0
        if similarity > best_score:
            best_score = similarity
            best_name = name
    if best_score > 0.42:
        return best_name, True
    return "Desconhecido", False

def correct_plate_characters(text):
    clean = "".join(e for e in text if e.isalnum()).upper()
    if len(clean) != 7:
        return clean
    l0 = 'O' if clean[0] == '0' else ('I' if clean[0] == '1' else clean[0])
    l1 = 'O' if clean[1] == '0' else ('I' if clean[1] == '1' else clean[1])
    l2 = 'O' if clean[2] == '0' else ('I' if clean[2] == '1' else clean[2])
    n3 = '0' if clean[3] == 'O' else ('1' if clean[3] == 'I' else clean[3])
    p4 = clean[4]
    if not p4.isdigit():
        p4 = '0' if p4 == 'O' else p4
    n5 = '0' if clean[5] == 'O' else ('1' if clean[5] == 'I' else clean[5])
    n6 = '0' if clean[6] == 'O' else ('1' if clean[6] == 'I' else clean[6])
    return f"{l0}{l1}{l2}{n3}{p4}{n5}{n6}"

# ==================== INTERFACE WEB (STREAMLIT) ====================

st.title("🚗 Sistema de Controle de Acesso - Portaria Inteligente")

# Menu de navegação lateral (Substitui os botões fixos de rodapé do Tkinter)
menu = st.sidebar.radio("Navegação", ["Identificação", "Cadastramento & Gestão"])

if menu == "Identificação":
    st.header("🔍 Painel de Identificação em Tempo Real")
    
    modo_id = st.radio("Escolha o Modo de Identificação:", ["Identificação de Carros (OCR)", "Identificação de Rostos"])
    
    # Utiliza a câmara nativa do navegador via Streamlit
    img_file_buffer = st.camera_input("Capturar Imagem da Câmara")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = frame.shape
        
        if modo_id == "Identificação de Carros (OCR)":
            st.info("A processar leitura de placa...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            results = ocr_reader.readtext(thresh)
            
            encontrou = False
            for (bbox, text, prob) in results:
                corrected_text = correct_plate_characters(text)
                if 6 <= len(corrected_text) <= 8 and prob > 0.25:
                    db_result = check_vehicle_database(corrected_text)
                    if db_result:
                        owner, status = db_result
                        if status == 'AUTHORIZED':
                            st.success(f"LIBERADO: Placa {corrected_text} ({owner})")
                        else:
                            st.error(f"BLOQUEADO: Placa {corrected_text} ({owner})")
                    else:
                        st.warning(f"NÃO CADASTRADO: Placa {corrected_text}")
                    encontrou = True
                    break
            if not encontrou:
                st.warning("Nenhuma placa válida detetada na imagem capturada.")
                
        else:  # Identificação de Rostos
            st.info("A processar reconhecimento facial...")
            face_roi = frame[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]
            if face_roi.size > 0:
                name, authorized = check_face_database(face_roi)
                if authorized:
                    st.success(f"LIBERADO: Rosto reconhecido de {name}")
                else:
                    st.error("DESCONHECIDO / NÃO CADASTRADO")
            else:
                st.error("Erro ao recortar a região do rosto.")

elif menu == "Cadastramento & Gestão":
    st.header("📋 Cadastramento & Gestão de Acessos")
    
    aba_gestao = st.tabs(["Gestão de Veículos", "Gestão de Rostos"])
    
    with aba_gestao[0]:
        st.subheader("Registar / Atualizar Veículo")
        p_plate = st.text_input("Placa do Veículo (ex: ABC1D23):").strip().upper()
        p_owner = st.text_input("Proprietário:").strip()
        p_status = st.selectbox("Status de Acesso:", ["AUTHORIZED", "BLOCKED"], format_func=lambda x: "Liberado" if x == "AUTHORIZED" else "Bloqueado")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💾 Salvar Placa"):
                if not p_plate or not p_owner:
                    st.warning("Preencha a placa e o proprietário do veículo!")
                else:
                    try:
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute('INSERT OR REPLACE INTO vehicles (plate, owner, status) VALUES (?, ?, ?)', (p_plate, p_owner, p_status))
                        conn.commit()
                        conn.close()
                        st.success(f"Veículo placa {p_plate} ({p_owner}) cadastrado com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao salvar a placa: {e}")
        with col_s2:
            if st.button("🗑️ Excluir Placa"):
                if not p_plate:
                    st.warning("Digite a placa do veículo que deseja remover!")
                else:
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM vehicles WHERE plate = ?', (p_plate,))
                    rows = cursor.rowcount
                    conn.commit()
                    conn.close()
                    if rows > 0:
                        st.success(f"Placa {p_plate} removida com sucesso!")
                    else:
                        st.error(f"A placa {p_plate} não foi encontrada no banco de dados.")

    with aba_gestao[1]:
        st.subheader("Registar / Excluir Rosto")
        f_name = st.text_input("Nome Completo para o Rosto:").strip()
        st.text("Nota: Para cadastrar o rosto, capture fotos utilizando a câmara abaixo:")
        f_foto = st.camera_input("Tirar Foto para Cadastro Facial")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if st.button("📸 Salvar Rosto Capturado"):
                if not f_name or f_foto is None:
                    st.warning("Digite o nome e tire uma foto antes de salvar!")
                else:
                    bytes_data = f_foto.getvalue()
                    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    h, w, _ = frame.shape
                    face_roi = frame[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]
                    
                    if face_roi.size > 0:
                        feat = extract_face_features(face_roi)
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        blob = pickle.dumps(feat)
                        cursor.execute('INSERT OR REPLACE INTO registered_faces (name, face_data) VALUES (?, ?)', (f_name, blob))
                        conn.commit()
                        conn.close()
                        st.success(f"Cadastro do rosto de '{f_name}' concluído com sucesso!")
                    else:
                        st.error("Falha ao recortar a região facial na imagem.")
        with col_f2:
            if st.button("🗑️ Excluir Rosto"):
                if not f_name:
                    st.warning("Digite o nome do cadastro que deseja remover!")
                else:
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM registered_faces WHERE name = ?', (f_name,))
                    rows = cursor.rowcount
                    conn.commit()
                    conn.close()
                    if rows > 0:
                        st.success(f"Cadastro de '{f_name}' removido com sucesso!")
                    else:
                        st.error(f"O nome '{f_name}' não foi encontrado no banco de dados.")