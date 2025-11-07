# movie_project.py
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QSpinBox, QComboBox,
    QTextEdit, QCheckBox, QPushButton, QVBoxLayout, QMessageBox
)
from PySide6.QtCore import Signal

class MovieProjectManager(QWidget):
    """
    Formulario + Gestor de proyecto de película/serie.
    Persiste en instancia, emite señal al guardar.
    """
    saved = Signal(dict)  # Se emite cuando el usuario guarda

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetflIA - Info del Proyecto")
        self.setGeometry(200, 200, 500, 700)
        self.setStyleSheet("background: #1e1e1e; color: #e0e0e0; font-family: Ubuntu;")

        # --- Datos del proyecto ---
        self.data = {
            "title": "",
            "duration": 90,
            "genre": "Acción",
            "is_series": False,
            "episodes": 1,
            "protagonists": 1,
            "prota_names": "",
            "extra_details": "",
            "final_type": "Feliz"
        }

        self.setup_ui()
        self.load_data()  # Carga datos guardados si existen

    def setup_ui(self):
        layout = QFormLayout()

        # Campos
        self.title_input = QLineEdit()
        self.duration_input = QSpinBox(); self.duration_input.setRange(1, 300); self.duration_input.setSuffix(" min")
        self.genre_input = QComboBox()
        self.genre_input.addItems([
            "Acción", "Drama", "Comedia", "Terror", "Ciencia Ficción",
            "Fantasía", "Suspenso", "Romance", "Documental", "Animación"
        ])
        self.is_series_input = QCheckBox("¿Es una serie?")
        self.episodes_input = QSpinBox(); self.episodes_input.setRange(1, 100); self.episodes_input.setEnabled(False)
        self.protagonists_input = QSpinBox(); self.protagonists_input.setRange(1, 10)
        self.prota_names_input = QTextEdit(); self.prota_names_input.setFixedHeight(60)
        self.extra_details_input = QTextEdit(); self.extra_details_input.setFixedHeight(80)
        self.final_type_input = QComboBox()
        self.final_type_input.addItems(["Feliz", "Triste", "Abierto", "Impactante", "Cíclico"])

        # Conexiones
        self.is_series_input.stateChanged.connect(self.toggle_episodes)

        # Botones
        btn_layout = QVBoxLayout()
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setStyleSheet("background: #ff4040; color: white; padding: 10px; border-radius: 8px;")
        self.cancel_btn.clicked.connect(self.close)

        self.save_btn = QPushButton("Guardar y Continuar")
        self.save_btn.setStyleSheet("background: #00bfff; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_data)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)

        # Añadir al layout
        layout.addRow("Título:", self.title_input)
        layout.addRow("Duración:", self.duration_input)
        layout.addRow("Género:", self.genre_input)
        layout.addRow(self.is_series_input)
        layout.addRow("Capítulos:", self.episodes_input)
        layout.addRow("Protagonistas:", self.protagonists_input)
        layout.addRow("Nombres (separados por coma):", self.prota_names_input)
        layout.addRow("Detalles extra (lugar, época, estilo):", self.extra_details_input)
        layout.addRow("Tipo de final:", self.final_type_input)
        layout.addRow(btn_layout)

        self.setLayout(layout)

    def toggle_episodes(self, state):
        self.episodes_input.setEnabled(state == 2)  # 2 = checked

    def load_data(self):
        """Carga datos desde self.data (persistente en instancia)"""
        self.title_input.setText(self.data["title"])
        self.duration_input.setValue(self.data["duration"])
        self.genre_input.setCurrentText(self.data["genre"])
        self.is_series_input.setChecked(self.data["is_series"])
        self.episodes_input.setValue(self.data["episodes"])
        self.protagonists_input.setValue(self.data["protagonists"])
        self.prota_names_input.setPlainText(self.data["prota_names"])
        self.extra_details_input.setPlainText(self.data["extra_details"])
        self.final_type_input.setCurrentText(self.data["final_type"])

    def save_data(self):
        """Guarda en memoria y emite señal"""
        if not self.title_input.text().strip():
            QMessageBox.warning(self, "Falta título", "El título es obligatorio.")
            return

        self.data.update({
            "title": self.title_input.text().strip(),
            "duration": self.duration_input.value(),
            "genre": self.genre_input.currentText(),
            "is_series": self.is_series_input.isChecked(),
            "episodes": self.episodes_input.value() if self.is_series_input.isChecked() else 1,
            "protagonists": self.protagonists_input.value(),
            "prota_names": self.prota_names_input.toPlainText().strip(),
            "extra_details": self.extra_details_input.toPlainText().strip(),
            "final_type": self.final_type_input.currentText()
        })

        print("[INFO] Proyecto guardado en memoria:", self.data)
        self.saved.emit(self.data.copy())  # Emite copia segura
        self.close()