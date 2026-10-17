import streamlit as st
import cv2
import sqlite3
import numpy as np
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(page_title="Portaria Inteligente", layout="wide")

st.title("🚗 Sistema de Controle de Acesso - Portaria Inteligente")

# Inicialização do Banco de Dados SQLite
def init_db():
    conn = sqlite3.connect("access_control.db")
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
    conn.commit()
    conn.close()

init_db()

# Barra Lateral - Navegação do Sistema
aba = st.sidebar.radio("Navegação", ["Identificação (Leitura)", "Cadastrar Rosto", "Cadastrar Veículo"])

if aba == "Identificação (Leitura)":
    st.header("🔍 Identificação de Acesso")
    
    img_file_buffer = st.camera_input("Posicione o Rosto ou a Placa na câmara")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Verificar Rosto", use_container_width=True):
                st.info("A processar reconhecimento facial...")
                # Insere aqui a tua lógica de reconhecimento facial (OpenCV / Dlib)
                
        with col2:
            if st.button("Verificar Placa (OCR)", use_container_width=True):
                st.info("A processar leitura de placa com EasyOCR...")
                # Insere aqui a tua lógica do EasyOCR com cv2_img

elif aba == "Cadastrar Rosto":
    st.header("📸 Cadastro de Rosto")
    nome = st.text_input("Nome Completo:")
    foto = st.camera_input("Tire uma foto para o cadastro facial")
    
    if st.button("Salvar Rosto") and nome and foto:
        # Aqui podes processar os dados da imagem e guardar no SQLite
        st.success(f"Rosto de {nome} cadastrado com sucesso!")

elif aba == "Cadastrar Veículo":
    st.header("🚗 Cadastro de Veículo")
    placa = st.text_input("Placa do Veículo (ex: ABC1D23):")
    proprietario = st.text_input("Nome do Proprietário:")
    status = st.selectbox("Status de Acesso", ["AUTHORIZED", "BLOCKED"])
    
    if st.button("Salvar Veículo") and placa and proprietario:
        conn = sqlite3.connect("access_control.db")
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO vehicles (plate, owner, status) VALUES (?, ?, ?)", 
                (placa.upper(), proprietario, status)
            )
            conn.commit()
            st.success(f"Veículo {placa.upper()} registado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao salvar no banco de dados: {e}")
        finally:
            conn.close()