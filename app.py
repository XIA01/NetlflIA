# app.py
import sys
import torch
import os
import gc
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QTextEdit, QPushButton, QProgressBar, QScrollArea,
    QGroupBox, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from diffusers import DiffusionPipeline
from PIL import Image

from pipelines.movie_project import MovieProjectManager
from pipelines.sinopsis import SinopsisGenerator

# =======================================================
#  CONFIGURACIÓN GLOBAL
# =======================================================
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =======================================================
#  HILO DE GENERACIÓN
# =======================================================
class GenerationThread(QThread):
    progress = Signal(int)
    finished = Signal(Image.Image)
    error = Signal(str)
    cancelled = False

    def __init__(self, prompt, pipe):
        super().__init__()
        self.prompt = prompt
        self.pipe = pipe

    def run(self):
        try:
            def callback(pipe, step: int, timestep: int, callback_kwargs):
                if self.cancelled:
                    raise KeyboardInterrupt()
                self.progress.emit(min(int(step / 28 * 100), 100))
                return callback_kwargs

            image = self.pipe(
                prompt=self.prompt,
                negative_prompt="borroso, feo, deformado, low quality, watermark, bad anatomy, extra limbs, text, ugly",
                num_inference_steps=28,
                guidance_scale=7.5,
                height=768,
                width=768,
                callback_on_step_end=callback
            ).images[0]

            self.finished.emit(image)
            gc.collect()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            self.error.emit("Cancelado por el usuario.")
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {str(e)}")

    def cancel(self):
        self.cancelled = True


# =======================================================
#  WIDGETS DE RESULTADOS
# =======================================================
class ResultWidget(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #00bfff; 
                border: 2px solid #333; 
                border-radius: 10px; 
                margin: 10px; 
                padding: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def add_text(self, text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #e0e0e0; margin: 5px;")
        self.layout.addWidget(label)


# =======================================================
#  MAIN WINDOW
# =======================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetflIA - Películas con IA Local")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background: #1a1a1a;")

        # === Layout principal ===
        central = QWidget()
        self.main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # === Columna Izquierda: Controles ===
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # Título
        title = QLabel("NetflIA - Generador de Películas")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00bfff; margin: 10px;")
        left_col.addWidget(title)

        # Botones secuenciales
        self.info_btn = self.create_step_btn("1. Info del Proyecto", self.open_project_form)
        left_col.addWidget(self.info_btn)

        self.sinopsis_btn = self.create_step_btn("2. Generar Sinopsis", None)
        self.sinopsis_btn.setEnabled(False)
        left_col.addWidget(self.sinopsis_btn)

        # Prompt + Generar
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Ej: Un gaucho en la luna, estilo realista...")
        self.prompt_input.setFixedHeight(80)
        self.prompt_input.setStyleSheet("background: #2d2d2d; color: white; padding: 10px; border-radius: 8px;")
        left_col.addWidget(self.prompt_input)

        self.load_model_btn = QPushButton("Cargar Modelos")
        self.load_model_btn.setStyleSheet(self.btn_style("f0ad4e"))
        self.load_model_btn.clicked.connect(self.load_model)
        left_col.addWidget(self.load_model_btn)

        self.generate_btn = QPushButton("Generar Imagen")
        self.generate_btn.setStyleSheet(self.btn_style("00bfff"))
        self.generate_btn.clicked.connect(self.start_generation)
        left_col.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(self.btn_style("ff4040"))
        self.cancel_btn.clicked.connect(self.cancel_generation)
        left_col.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar::chunk { background: #00bfff; }")
        left_col.addWidget(self.progress)

        # === Columna Derecha: Resultados (Scroll) ===
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_area.setStyleSheet("background: #222; border: none;")
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_area.setWidget(self.results_widget)

        # Añadir columnas
        self.main_layout.addLayout(left_col, 1)
        self.main_layout.addWidget(self.results_area, 2)

        # === Cargar modelo ===
        self.pipe = None
        self.thread = None
        self.project_manager = None
        self.project_data = None

    # =======================================================
    #  UTILIDADES
    # =======================================================
    def create_step_btn(self, text, callback):
        btn = QPushButton(text)
        btn.setStyleSheet(self.btn_style("00bfff"))
        if callback:
            btn.clicked.connect(callback)
        return btn

    def btn_style(self, color):
        return f"""
            QPushButton {{ background: #{color}; color: white; padding: 12px; border-radius: 8px; font-weight: bold; }}
            QPushButton:disabled {{ background: #555; color: #aaa; }}
            QPushButton:hover:!disabled {{ background: #{color}dd; }}
        """

    # =======================================================
    #  MODELO
    # =======================================================
    def load_model(self):
        try:
            self.add_result("Cargando SDXL...", "Cargando modelo...")
            QApplication.processEvents()

            self.pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            )
            self.pipe.to("cuda")
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except: pass

            self.add_result("Modelo cargado", "Listo para generar.")
        except Exception as e:
            self.add_result("Error", str(e))

    # =======================================================
    #  RESULTADOS
    # =======================================================
    def add_result(self, title, content):
        widget = ResultWidget(title)
        widget.add_text(content)
        self.results_layout.addWidget(widget)
        self.results_area.verticalScrollBar().setValue(self.results_area.verticalScrollBar().maximum())

    def clear_results(self):
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().deleteLater()

    # =======================================================
    #  PASO 1: FORMULARIO
    # =======================================================
    def open_project_form(self):
        if not self.project_manager:
            self.project_manager = MovieProjectManager()
            self.project_manager.saved.connect(self.on_project_saved)
        self.project_manager.show()

    def on_project_saved(self, data):
        self.project_data = data
        self.sinopsis_btn.setEnabled(True)
        self.sinopsis_btn.clicked.connect(self.generate_sinopsis)

        # Mostrar datos en columna derecha
        self.add_result("Sinopsis - Datos del Proyecto", self.format_project_data(data))

    def format_project_data(self, data):
        lines = [
            f"<b>Título:</b> {data['title']}",
            f"<b>Duración:</b> {data['duration']} min",
            f"<b>Género:</b> {data['genre']}",
            f"<b>Tipo:</b> {'Serie' if data['is_series'] else 'Película'}",
        ]
        if data['is_series']:
            lines.append(f"<b>Capítulos:</b> {data['episodes']}")
        lines.append(f"<b>Protagonistas:</b> {data['prota_names'] or 'No especificados'}")
        lines.append(f"<b>Final:</b> {data['final_type']}")
        if data['extra_details']:
            lines.append(f"<b>Detalles:</b> {data['extra_details']}")
        return "<br>".join(lines)

    # =======================================================
    #  PASO 2: SINOPSIS
    # =======================================================
    def generate_sinopsis(self):
        if not self.project_data:
            return
        self.sinopsis_btn.setEnabled(False)
        self.add_result("Sinopsis", "Generando con Qwen-7B...")

        # Aquí irá vLLM
        sinopsis = SinopsisGenerator().generate(self.project_data)
        # sinopsis = "Sinopsis generada automáticamente..."  # ← Placeholder

        # Reemplazar último widget
        last = self.results_layout.itemAt(self.results_layout.count() - 1).widget()
        last.deleteLater()
        widget = ResultWidget("Sinopsis")
        widget.add_text(sinopsis)
        regen_btn = QPushButton("Regenerar Sinopsis")
        regen_btn.setStyleSheet(self.btn_style("ff6b6b"))
        regen_btn.clicked.connect(self.generate_sinopsis)
        widget.layout.addWidget(regen_btn)
        self.results_layout.addWidget(widget)
        self.sinopsis_btn.setEnabled(True)

    # =======================================================
    #  GENERACIÓN DE IMAGEN
    # =======================================================
    def start_generation(self):
        if not self.pipe:
            QMessageBox.warning(self, "Error", "Por favor, carga los modelos antes de generar una imagen.")
            return

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Error", "Escribe un prompt.")
            return
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.thread = GenerationThread(prompt, self.pipe)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.show_image)
        self.thread.error.connect(lambda e: self.add_result("Error", e))
        self.thread.start()

    def cancel_generation(self):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.cancel_btn.setEnabled(False)

    def show_image(self, img: Image.Image):
        path = "ultima_generacion.png"
        img.save(path)
        pixmap = QPixmap(path).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        widget = ResultWidget("Imagen Generada")
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        widget.layout.addWidget(label)
        self.results_layout.addWidget(widget)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)


# =======================================================
#  MAIN
# =======================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())