# pipelines/sinopsis.py
from openai import OpenAI
import logging

# Configurar logging (opcional, para depuración)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SinopsisGenerator:
    def __init__(self):
        # NO inicializar cliente aquí → lazy load
        self.client = None
        self.base_url = "http://localhost:8001/v1"
        self.api_key = "none"
        self.model = "Qwen/Qwen2.5-3B-Instruct"  # Ajusta si usas otro

    def _init_client(self):
        """Inicializa el cliente solo cuando se necesita"""
        if self.client is None:
            try:
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                logger.info("Cliente vLLM inicializado correctamente")
            except Exception as e:
                logger.error(f"Error conectando a vLLM: {e}")
                self.client = None

    def generate(self, data: dict) -> str:
        """
        Genera sinopsis usando vLLM (Qwen local).
        Solo intenta conectar cuando se llama.
        """
        # Inicializar cliente lazy
        self._init_client()
        if not self.client:
            return "Error: vLLM no está corriendo en http://localhost:8001"

        prompt = f"""
        Escribe una sinopsis atractiva y profesional de {data['duration']} minutos para una {'serie' if data['is_series'] else 'película'} de {data['genre']}.
       
        Título: {data['title']}
        Protagonistas: {', '.join([p.strip() for p in data['prota_names'].split(',')[:3]]) if data['prota_names'] else 'Un protagonista'}
        Detalles adicionales: {data['extra_details'] or 'Ninguno'}
        Tipo de final: {data['final_type']}
       
        Máximo 3 párrafos. Estilo cinematográfico, impactante y claro.
        """.strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                return "Error: vLLM no responde. Asegúrate de que esté corriendo en el puerto 8001."
            elif "timeout" in error_msg.lower():
                return "Error: Tiempo agotado. vLLM está lento o sobrecargado."
            else:
                return f"Error de vLLM: {error_msg}"