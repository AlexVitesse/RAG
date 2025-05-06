"""
Chat con Transcripción de Voz - Aplicación Streamlit
Permite interactuar mediante texto o voz utilizando streamlit_mic_recorder.

FEATURES:
- TRANSCRIPCIÓN DE VOZ CON streamlit_mic_recorder
- INTERACCION CON LLM (GROQ) MODELO Llama3-8B
- RESPUESTAS DE AUDIO CON EDGE_TTS
- MEDICIÓN DE TIEMPOS
- MEDICION DE TOKENS
- UN POCO MODULOS DE LLM (GROQ) MODELO Llama3-8B
"""
import streamlit as st
from streamlit_mic_recorder import speech_to_text
import time
import edge_tts
import asyncio
from io import BytesIO
import base64
from datetime import datetime
from rag import ask_question


# Constantes
APP_TITLE = "💬 Platica con la IA   "
APP_ICON = "💬"
PAGE_TITLE = "Chat Simple"
DEFAULT_LANGUAGE = 'es'
THINKING_DELAY = 1
VOICE = "es-MX-DaliaNeural"  # Voz Dalia en español mexicano


def initialize_app():
    """Configura la página y verifica el estado inicial."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=APP_ICON)
    st.title(APP_TITLE)
    
    # Inicializar el historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Muestra el historial de mensajes existente."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Mostrar el reproductor de audio si es una respuesta del asistente
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format='audio/wav')

async def generate_audio(text, voice=VOICE, rate="+0%", pitch="+0Hz", volume="+0%"):
    """Genera audio con parámetros en formato EXACTO como requiere edge_tts"""
    try:
        start_time = datetime.now()
        print(f"[{start_time}] Iniciando generación de audio...")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
        
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time}] Audio generado en {duration:.2f} segundos")
        
        return bytes(audio_data) if audio_data else None
    
    except Exception as e:
        st.error(f"Error en generación de audio: {str(e)}")
        return None

def create_input_area():
    """Crea el área de entrada con opciones de texto y voz."""
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            text_input = st.chat_input(
                placeholder="Escribe tu mensaje aquí...",
                key="text_input"
            )
        with col2:
            voice_input = speech_to_text(
                language=DEFAULT_LANGUAGE,
                start_prompt="🎤",
                stop_prompt="⏹️",
                just_once=True,
                use_container_width=True,
                key="speech_to_text"
            )
    return voice_input if voice_input else text_input

def generate_response(user_message):
    """
    Genera una respuesta basada en el mensaje del usuario.
    Mide el tiempo que tarda la respuesta del LLM.
    
    Args:
        user_message (str): El mensaje del usuario
    
    Returns:
        str: La respuesta generada
    """
    start_time = datetime.now()
    print(f"[{start_time}] Consultando al LLM...")
    response = ask_question(user_message)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"[{end_time}] Respuesta del LLM recibida en {duration:.2f} segundos")
    return response

async def add_message(role, content):
    """
    Añade un mensaje al historial y lo muestra.
    Si es del asistente, genera y muestra audio.
    
    Args:
        role (str): El rol del mensaje ('user' o 'assistant')
        content (str): El contenido del mensaje
    """
    message_data = {"role": role, "content": content}
    
    if role == "assistant":
        # Generar audio para la respuesta
        audio_data = await generate_audio(content)
        if audio_data:
            message_data["audio"] = audio_data
    
    st.session_state.messages.append(message_data)
    with st.chat_message(role):
        st.write(content)
        if role == "assistant" and "audio" in message_data:
            st.audio(message_data["audio"], format='audio/wav')

async def process_user_input(user_input):
    """
    Procesa el input del usuario y genera una respuesta.
    Mide el tiempo total del proceso.
    
    Args:
        user_input (str): El mensaje del usuario
    """
    total_start = datetime.now()
    print(f"\n[{total_start}] Iniciando procesamiento de mensaje...")
    
    # Agregar y mostrar mensaje del usuario
    await add_message("user", user_input)
    
    # Generar y mostrar respuesta
    with st.spinner("Pensando..."):
        time.sleep(THINKING_DELAY)  # Simular tiempo de procesamiento
        
        # Medir tiempo de generación de respuesta
        llm_start = time.time()
        response = generate_response(user_input)
        llm_duration = time.time() - llm_start
        
        # Medir tiempo de generación de audio
        audio_start = time.time()
        await add_message("assistant", response)
        audio_duration = time.time() - audio_start
        
        # Calcular tiempo total
        total_duration = (datetime.now() - total_start).total_seconds()
        
        # Mostrar métricas en consola
        print("\n--- Métricas de Tiempo ---")
        print(f"LLM: {llm_duration:.2f} segundos")
        print(f"Audio: {audio_duration:.2f} segundos")
        print(f"Total: {total_duration:.2f} segundos")
        print("--------------------------\n")
    
    # Actualizar la interfaz
    st.rerun()

def display_sidebar_instructions():
    """Muestra instrucciones de uso en la barra lateral."""
    st.sidebar.markdown("""
    ### Cómo usar:
    1. Escribe o habla con 🎤
    2. Envía con Enter/⏹️
    3. Recibe respuestas en texto y audio
    
    ### Funcionalidades:
    - Transcripción de voz
    - Respuestas en audio (voz de Dalia)
    - Historial de chat persistente
    - Respuestas instantáneas
    """)

async def main():
    """Función principal que ejecuta la aplicación."""
    initialize_app()
    display_chat_history()
    display_sidebar_instructions()
    
    # Capturar entrada del usuario
    user_input = create_input_area()
    
    # Procesar entrada si existe
    if user_input:
        await process_user_input(user_input)

if __name__ == "__main__":
    asyncio.run(main())