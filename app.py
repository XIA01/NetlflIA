import sys
import torch
import traceback
import os
import gc
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QProgressBar, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from diffusers import DiffusionPipeline
from PIL import Image

# =======================================================
#  🔧 CONFIGURACIÓN GLOBAL
# =======================================================
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =======================================================
#  🧩 HILO DE GENERACIÓN
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
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA no disponible en el sistema.")

            # Callback de progreso
            def callback(pipe, step: int, timestep: int, callback_kwargs):
                if self.cancelled:
                    raise KeyboardInterrupt("Generación cancelada por el usuario.")
                progress = min(int(step / 28 * 100), 100)
                self.progress.emit(progress)
                return callback_kwargs

            # Generar imagen
            image = self.pipe(
                prompt=self.prompt,
                negative_prompt="borroso, feo, deformado, low quality, watermark, bad anatomy, extra limbs, text, ugly",
                num_inference_steps=28,
                guidance_scale=7.5,
                height=768,
                width=768,
                callback_on_step_end=callback
            ).images[0]

            # Emitir antes de limpiar
            self.finished.emit(image)

            # Limpieza ligera
            gc.collect()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("[INFO] Generación cancelada.")
            self.error.emit("Cancelado por el usuario.")
        except Exception as e:
            print(traceback.format_exc())
            self.error.emit(f"{type(e).__name__}: {str(e)}")

    def cancel(self):
        self.cancelled = True


# =======================================================
#  🖥️ INTERFAZ PRINCIPAL
# =======================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetflIA - SDXL Local @Como_Critica")
        self.setGeometry(100, 100, 720, 820)
        self.setStyleSheet("background: #1a1a1a; color: #e0e0e0; font-family: Ubuntu, Arial;")

        layout = QVBoxLayout()

        title = QLabel("Generador IA Local - RTX 3060")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px; color: #00bfff;")
        layout.addWidget(title)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Ej: Un mate con vapor, mesa de madera, luz natural, hiperrealista...")
        self.prompt_input.setStyleSheet("background: #2d2d2d; padding: 12px; border-radius: 8px; font-size: 14px;")
        self.prompt_input.setFixedHeight(80)
        layout.addWidget(self.prompt_input)

        # Botones
        self.generate_btn = QPushButton("Generar Imagen")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background: #00bfff; color: white; padding: 14px;
                border-radius: 10px; font-weight: bold; font-size: 16px;
            }
            QPushButton:disabled { background: #555; }
            QPushButton:hover:!disabled { background: #00a0d4; }
        """)
        self.generate_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("background: #ff4040; color: white; border-radius: 8px; padding: 10px;")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        layout.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar { border-radius: 8px; text-align: center; }
            QProgressBar::chunk { background: #00bfff; border-radius: 8px; }
        """)
        layout.addWidget(self.progress)

        self.image_label = QLabel("→ Escribe un prompt y haz clic en 'Generar'")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #252525; border: 2px dashed #444; border-radius: 12px; margin: 10px; padding: 20px;")
        self.image_label.setMinimumHeight(420)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # Cargar modelo una sola vez
        self.pipe = None
        self.load_model()

        self.thread = None

    # =======================================================
    #  CARGA DEL MODELO
    # =======================================================
    def load_model(self):
        try:
            self.image_label.setText("Cargando modelo SDXL... (puede tardar unos segundos)")
            QApplication.processEvents()

            self.pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            self.pipe.to("cuda")

            # Optimización de memoria
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[INFO] xFormers habilitado.")
            except Exception:
                print("[AVISO] xFormers no disponible.")

            self.image_label.setText("✅ Modelo cargado. Escribe un prompt y genera.")
        except Exception as e:
            self.image_label.setText(f"Error al cargar modelo: {e}")

    # =======================================================
    #  GENERACIÓN
    # =======================================================
    def start_generation(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Falta prompt", "Escribe algo para generar.")
            return

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.image_label.setText("🧠 Generando...")

        self.thread = GenerationThread(prompt, self.pipe)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.show_image)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def cancel_generation(self):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.cancel_btn.setEnabled(False)

    # =======================================================
    #  RESULTADOS
    # =======================================================
    def show_image(self, img: Image.Image):
        output_path = "ultima_generacion.png"
        img.save(output_path)
        pixmap = QPixmap(output_path).scaled(
            660, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100)

        torch.cuda.empty_cache()

    def show_error(self, error_msg: str):
        self.image_label.setText(f"⚠️ {error_msg}")
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(0)
        print(f"[GUI Error] {error_msg}")
        torch.cuda.empty_cache()


# =======================================================
#  MAIN
# =======================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
