import cv2
import sqlite3
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import pickle
import easyocr
import re

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, bg_color, fg_color, hover_color, command, radius=20, width=220, height=42):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.radius = radius
        self.width = width
        self.height = height

        self.rect = self.draw_rounded_rect(0, 0, width, height, radius, fill=bg_color)
        self.text_id = self.create_text(width / 2, height / 2, text=text, fill=fg_color, font=("Segoe UI", 10, "bold"))

        self.bind("<Button-1>", lambda e: self.command())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.config(cursor="hand2")

    def draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)

    def on_leave(self, event):
        self.itemconfig(self.rect, fill=self.bg_color)


class RoundedEntry(tk.Canvas):
    def __init__(self, parent, width=220, height=34, bg_color="#FFFFFF", fg_color="#0F172A", radius=16):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.width = width
        self.height = height
        self.radius = radius
        self.bg_color = bg_color

        self.bg_rect = self.draw_rounded_rect(0, 0, width, height, radius, fill=bg_color, outline="#CBD5E1", width=1)
        self.entry = tk.Entry(self, bg=bg_color, fg=fg_color, font=("Segoe UI", 10), bd=0, highlightthickness=0)
        self.create_window(width / 2, height / 2, window=self.entry, width=width - 16, height=height - 8)

    def draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def get(self):
        return self.entry.get()

    def delete(self, first, last=tk.END):
        self.entry.delete(first, last)


class AccessControlApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1150x800")
        
        self.window.configure(bg="#F8FAFC")

        self.db_name = "access_control.db"
        self.init_database()

        print("[INFO] Carregando modelo OCR... Aguarde.")
        self.ocr_reader = easyocr.Reader(['pt', 'en'], gpu=False)

        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_mode = None
        self.lock = threading.Lock()

        self.active_alert_text = ""
        self.active_alert_bg_color = (0, 0, 0)
        self.alert_expiration_time = 0
        self.capture_instruction = ""

        self.create_widgets()

        self.window.bind("<Escape>", self.on_key_press)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_database(self):
        conn = sqlite3.connect(self.db_name)
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

    def check_vehicle_database(self, plate):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT owner, status FROM vehicles WHERE plate = ?', (plate.upper(),))
        result = cursor.fetchone()
        conn.close()
        return result

    def extract_face_features(self, face_roi):
        """ Extrai vetor de textura por matriz de blocos 4x4 pré-processados """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        
        # Filtro CLAHE para normalizar sombra/luz
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Divisão em blocos 4x4 (16 sub-regiões)
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

    def check_face_database(self, face_roi):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT name, face_data FROM registered_faces')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Desconhecido", False

        current_features = self.extract_face_features(face_roi)

        best_score = -1.0
        best_name = "Desconhecido"

        for name, blob in rows:
            saved_features = pickle.loads(blob)
            
            # Se for array de recursos extraídos no novo formato
            if isinstance(saved_features, np.ndarray) and len(saved_features) == len(current_features):
                # Distância por Correlação entre vetores
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

        # Tolerância realista de leitura (> 0.42)
        if best_score > 0.42:
            return best_name, True
        return "Desconhecido", False

    def create_widgets(self):
        self.title_label = tk.Label(
            self.window, text="Sistema de Controle de Acesso - Portaria Inteligente",
            font=("Segoe UI", 16, "bold"), fg="#0F172A", bg="#F8FAFC"
        )
        self.title_label.pack(pady=8)

        btn_frame = tk.Frame(self.window, bg="#F8FAFC")
        btn_frame.pack(side=tk.BOTTOM, pady=12)

        btn_rosto = RoundedButton(
            btn_frame, text="Identificação de Rosto", bg_color="#0F172A", fg_color="#FFFFFF",
            hover_color="#1E293B", command=self.start_face_recognition, radius=20, width=220, height=42
        )
        btn_rosto.grid(row=0, column=0, padx=12)

        btn_carros = RoundedButton(
            btn_frame, text="Identificação de Carros", bg_color="#FF6B00", fg_color="#FFFFFF",
            hover_color="#E05E00", command=self.start_car_recognition, radius=20, width=220, height=42
        )
        btn_carros.grid(row=0, column=1, padx=12)

        btn_stop = RoundedButton(
            btn_frame, text="⏹️ Parar / Sair (ESC)", bg_color="#64748B", fg_color="#FFFFFF",
            hover_color="#475569", command=self.stop_source, radius=20, width=180, height=42
        )
        btn_stop.grid(row=0, column=2, padx=12)

        main_content_frame = tk.Frame(self.window, bg="#F8FAFC")
        main_content_frame.pack(pady=5, fill=tk.BOTH, expand=True, padx=20)

        self.video_container = tk.Frame(main_content_frame, bg="#0F172A", padx=0, pady=0)
        self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_container.pack_propagate(False)

        self.video_label = tk.Label(self.video_container, bg="#0F172A")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        cadastramento_frame = tk.LabelFrame(
            main_content_frame, text=" Cadastramento & Gestão ", font=("Segoe UI", 11, "bold"),
            fg="#0F172A", bg="#E2E8F0", padx=15, pady=15, width=330
        )
        cadastramento_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        cadastramento_frame.pack_propagate(False)

        lbl_face = tk.Label(cadastramento_frame, text="Gestão de Rosto:", font=("Segoe UI", 10, "bold"), fg="#0F172A", bg="#E2E8F0")
        lbl_face.pack(anchor="w", pady=(5, 2))

        tk.Label(cadastramento_frame, text="Nome Pessoa:", font=("Segoe UI", 9), fg="#0F172A", bg="#E2E8F0").pack(anchor="w")
        self.face_name_entry = RoundedEntry(cadastramento_frame, width=300, height=34, radius=16)
        self.face_name_entry.pack(fill=tk.X, pady=(2, 6))

        btn_face_box = tk.Frame(cadastramento_frame, bg="#E2E8F0")
        btn_face_box.pack(fill=tk.X, pady=(0, 15))

        self.btn_capture_face = RoundedButton(
            btn_face_box, text="📸 Cadastrar Rosto", bg_color="#2563EB", fg_color="#FFFFFF",
            hover_color="#1D4ED8", command=self.capture_face_samples, radius=18, width=142, height=38
        )
        self.btn_capture_face.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_delete_face = RoundedButton(
            btn_face_box, text="🗑️ Excluir Rosto", bg_color="#DC2626", fg_color="#FFFFFF",
            hover_color="#B91C1C", command=self.delete_face_from_database, radius=18, width=142, height=38
        )
        self.btn_delete_face.pack(side=tk.RIGHT, padx=(4, 0))

        tk.Frame(cadastramento_frame, height=2, bg="#CBD5E1").pack(fill=tk.X, pady=8)

        lbl_car = tk.Label(cadastramento_frame, text="Gestão de Carro / Veículo:", font=("Segoe UI", 10, "bold"), fg="#0F172A", bg="#E2E8F0")
        lbl_car.pack(anchor="w", pady=(5, 2))

        tk.Label(cadastramento_frame, text="Placa:", font=("Segoe UI", 9), fg="#0F172A", bg="#E2E8F0").pack(anchor="w")
        self.plate_entry = RoundedEntry(cadastramento_frame, width=300, height=34, radius=16)
        self.plate_entry.pack(fill=tk.X, pady=(2, 6))

        tk.Label(cadastramento_frame, text="Proprietário:", font=("Segoe UI", 9), fg="#0F172A", bg="#E2E8F0").pack(anchor="w")
        self.owner_entry = RoundedEntry(cadastramento_frame, width=300, height=34, radius=16)
        self.owner_entry.pack(fill=tk.X, pady=(2, 6))

        tk.Label(cadastramento_frame, text="Status de Acesso:", font=("Segoe UI", 9), fg="#0F172A", bg="#E2E8F0").pack(anchor="w")
        
        status_frame = tk.Frame(cadastramento_frame, bg="#E2E8F0")
        status_frame.pack(fill=tk.X, pady=(2, 8))

        self.status_var = tk.StringVar(value="AUTHORIZED")
        tk.Radiobutton(status_frame, text="Liberado", variable=self.status_var, value="AUTHORIZED", bg="#E2E8F0", font=("Segoe UI", 9, "bold"), fg="#166534").pack(side=tk.LEFT)
        tk.Radiobutton(status_frame, text="Bloqueado", variable=self.status_var, value="BLOCKED", bg="#E2E8F0", font=("Segoe UI", 9, "bold"), fg="#991B1B").pack(side=tk.LEFT, padx=10)

        btn_car_box = tk.Frame(cadastramento_frame, bg="#E2E8F0")
        btn_car_box.pack(fill=tk.X)

        self.btn_save_car = RoundedButton(
            btn_car_box, text="💾 Salvar Placa", bg_color="#0F172A", fg_color="#FFFFFF",
            hover_color="#1E293B", command=self.save_car_to_database, radius=18, width=142, height=38
        )
        self.btn_save_car.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_delete_car = RoundedButton(
            btn_car_box, text="🗑️ Excluir Placa", bg_color="#DC2626", fg_color="#FFFFFF",
            hover_color="#B91C1C", command=self.delete_car_from_database, radius=18, width=142, height=38
        )
        self.btn_delete_car.pack(side=tk.RIGHT, padx=(4, 0))

    def save_car_to_database(self):
        plate = self.plate_entry.get().strip().upper()
        owner = self.owner_entry.get().strip()
        status = self.status_var.get()

        if not plate or not owner:
            messagebox.showwarning("Aviso", "Preencha a placa e o proprietário do veículo!")
            return

        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO vehicles (plate, owner, status) VALUES (?, ?, ?)', (plate, owner, status))
            conn.commit()
            conn.close()
            print("\a")
            messagebox.showinfo("Sucesso", f"Veículo placa {plate} ({owner}) cadastrado com sucesso!")
            self.plate_entry.delete(0, tk.END)
            self.owner_entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar a placa: {e}")

    def delete_car_from_database(self):
        plate = self.plate_entry.get().strip().upper()
        if not plate:
            messagebox.showwarning("Aviso", "Digite a placa do veículo que deseja descadastrar!")
            return

        if messagebox.askyesno("Confirmar Exclusão", f"Tem certeza que deseja remover o veículo de placa {plate}?"):
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM vehicles WHERE plate = ?', (plate,))
            rows = cursor.rowcount
            conn.commit()
            conn.close()

            if rows > 0:
                print("\a")
                messagebox.showinfo("Sucesso", f"Placa {plate} removida com sucesso!")
                self.plate_entry.delete(0, tk.END)
                self.owner_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Erro", f"A placa {plate} não foi encontrada no banco de dados.")

    def capture_face_samples(self):
        name = self.face_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Aviso", "Digite o nome da pessoa antes de iniciar o cadastro de rosto!")
            return

        self.stop_source()
        threading.Thread(target=self._run_capture_process, args=(name,), daemon=True).start()

    def _run_capture_process(self, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
            return

        all_features = []
        stages = [
            ("1/3: Vire ligeiramente para a ESQUERDA", 10),
            ("2/3: Olhe para a FRENTE", 10),
            ("3/3: Vire ligeiramente para a DIREITA", 10)
        ]

        for instruction, count_target in stages:
            self.capture_instruction = instruction
            current_count = 0
            
            while current_count < count_target:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                face_roi = frame[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]
                if face_roi.size == 0:
                    continue

                feat = self.extract_face_features(face_roi)
                all_features.append(feat)
                
                current_count += 1
                time.sleep(0.25)

        cap.release()
        self.capture_instruction = ""

        if len(all_features) >= 20:
            # Média das matrizes de textura do rosto
            mean_feature_vector = np.mean(all_features, axis=0)
            
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            blob = pickle.dumps(mean_feature_vector)
            cursor.execute('INSERT OR REPLACE INTO registered_faces (name, face_data) VALUES (?, ?)', (name, blob))
            conn.commit()
            conn.close()

            print("\a")
            messagebox.showinfo("✅ Sucesso!", f"Recadastro do rosto de '{name}' concluído com sucesso!")
            self.face_name_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Erro", "Falha na captura das fotos. Tente novamente.")

    def delete_face_from_database(self):
        name = self.face_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Aviso", "Digite o nome da pessoa que deseja descadastrar!")
            return

        if messagebox.askyesno("Confirmar Exclusão", f"Tem certeza que deseja remover o cadastro do rosto de '{name}'?"):
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM registered_faces WHERE name = ?', (name,))
            rows = cursor.rowcount
            conn.commit()
            conn.close()

            if rows > 0:
                print("\a")
                messagebox.showinfo("Sucesso", f"Cadastro de '{name}' removido com sucesso!")
                self.face_name_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Erro", f"O nome '{name}' não foi encontrado no banco de dados.")

    def trigger_alert(self, text, bg_color):
        if time.time() >= self.alert_expiration_time:
            self.active_alert_text = text
            self.active_alert_bg_color = bg_color
            self.alert_expiration_time = time.time() + 5.0

    def correct_plate_characters(self, text):
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

    def ocr_worker(self):
        while self.running and self.current_mode == 'car_id':
            if self.current_frame is None:
                time.sleep(0.01)
                continue

            if time.time() < self.alert_expiration_time:
                time.sleep(0.5)
                continue

            with self.lock:
                frame_to_process = self.current_frame.copy()

            small_frame = cv2.resize(frame_to_process, (0, 0), fx=1.0, fy=1.0)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            results = self.ocr_reader.readtext(thresh)

            found_valid = False
            for (bbox, text, prob) in results:
                corrected_text = self.correct_plate_characters(text)
                
                if 6 <= len(corrected_text) <= 8 and prob > 0.25:
                    db_result = self.check_vehicle_database(corrected_text)

                    if db_result:
                        owner, status = db_result
                        if status == 'AUTHORIZED':
                            self.trigger_alert(f"LIBERADO: Placa {corrected_text} ({owner})", (0, 160, 0))
                        else:
                            self.trigger_alert(f"BLOQUEADO: Placa {corrected_text} ({owner})", (0, 0, 200))
                    else:
                        self.trigger_alert(f"NAO CADASTRADO: Placa {corrected_text}", (0, 120, 220))
                    
                    found_valid = True
                    break

            if found_valid:
                time.sleep(1.0)
            else:
                time.sleep(0.3)

    def face_worker(self):
        while self.running and self.current_mode == 'face_id':
            if self.current_frame is None:
                time.sleep(0.01)
                continue

            if time.time() < self.alert_expiration_time:
                time.sleep(0.5)
                continue

            with self.lock:
                frame_to_process = self.current_frame.copy()

            h, w, _ = frame_to_process.shape
            face_roi = frame_to_process[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]
            if face_roi.size == 0:
                continue

            name, authorized = self.check_face_database(face_roi)

            if authorized:
                self.trigger_alert(f"LIBERADO: {name}", (0, 160, 0))
            else:
                self.trigger_alert(f"DESCONHECIDO / NAO CADASTRADO", (0, 0, 200))

            time.sleep(1.0)

    def start_source(self, source, mode):
        self.stop_source()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", f"Não foi possível abrir a fonte: {source}")
            return

        self.running = True
        self.current_mode = mode
        self.alert_expiration_time = 0

        if mode == 'car_id':
            self.worker_thread = threading.Thread(target=self.ocr_worker, daemon=True)
            self.title_label.config(text="Identificação - Carros (Webcam Ativa)")
        else:
            self.worker_thread = threading.Thread(target=self.face_worker, daemon=True)
            self.title_label.config(text="Identificação - Rostos (Webcam Ativa)")
            
        self.worker_thread.start()
        self.update_frame()

    def start_face_recognition(self):
        self.start_source(0, 'face_id')

    def start_car_recognition(self):
        self.start_source(0, 'car_id')

    def update_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
                    
                # Exibe orientação na tela durante o cadastro das 30 fotos
                if self.capture_instruction:
                    overlay = frame.copy()
                    h_frame, w_frame, _ = frame.shape
                    cv2.rectangle(overlay, (20, 20), (w_frame - 20, 90), (0, 100, 255), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, self.capture_instruction, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                current_time = time.time()
                if current_time < self.alert_expiration_time:
                    remaining = int(self.alert_expiration_time - current_time) + 1
                    display_text = f"{self.active_alert_text}  [{remaining}s]"
                    
                    overlay = frame.copy()
                    h_frame, w_frame, _ = frame.shape
                    
                    cv2.rectangle(overlay, (25, 20), (w_frame - 25, 85), self.active_alert_bg_color, -1)
                    
                    alpha = 0.55
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                    cv2.rectangle(frame, (25, 20), (w_frame - 25, 85), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, display_text, (45, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if len(frame.shape) == 2 else cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                container_w = self.video_container.winfo_width()
                container_h = self.video_container.winfo_height()

                if container_w > 50 and container_h > 50:
                    orig_w, orig_h = img.size
                    ratio = min(container_w / orig_w, container_h / orig_h)
                    new_w = int(orig_w * ratio)
                    new_h = int(orig_h * ratio)
                    
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.window.after(10, self.update_frame)
            else:
                self.stop_source()

    def stop_source(self):
        self.running = False
        self.current_mode = None
        self.capture_instruction = ""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.title_label.config(text="Sistema de Controle de Acesso - Portaria Inteligente")

    def on_key_press(self, event):
        self.on_closing()

    def on_closing(self):
        self.stop_source()
        self.window.destroy()

if __name__ == "__main__":
    print("=" * 60)
    print(" Para rodar o programa, copie e cole o comando abaixo:")
    print(" python Entrada.de.carros.py")
    print("=" * 60)
    
    root = tk.Tk()
    app = AccessControlApp(root, "Painel de Controle de Acesso - Portaria Inteligente")
    root.mainloop()