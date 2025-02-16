# app.py
# This app is a voice-enabled version of the cooking assistant app
# app uses pinecone for vector database

import os
import streamlit as st
from backend.doc_processor import DocProcessor
from backend.response_generator import ResponseGenerator
import speech_recognition as sr
import pyttsx3
import base64
import tempfile

# Constants
# PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "models/embedding-001"

def initialize_session_state():
    """Initialize all necessary session state variables."""
    if "vector_retriever" not in st.session_state:
        st.session_state.vector_retriever = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = False
    if "selected_voice" not in st.session_state:
        st.session_state.selected_voice = "Girl"

def recognize_speech():
    """Convert speech to text using speech_recognition."""
    recognizer = sr.Recognizer()
    try:
        st.toast("Listening...", icon="🎤")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Increase timeout and phrase_time_limit to capture longer sentences
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        st.error("Could not understand audio. Please try again.")
        return None
    except sr.WaitTimeoutError:
        # Log the timeout error without disrupting the UI
        print("Listening timed out while waiting for phrase to start.")
        return None
    except sr.RequestError as e:
        st.error(f"Error: {str(e)}")
        return None

def text_to_speech(text):
    """
    Convert text to speech using pyttsx3.
    This function selects a voice based on the user's choice ("Girl" or "Boy")
    and saves the output to a temporary WAV file which is then encoded in base64.
    """
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_choice = st.session_state.get("selected_voice", "Girl").lower()

        # Choose a voice based on the selection.
        # On many Windows systems, voices[0] is male and voices[1] is female.
        if voice_choice == "girl" and len(voices) > 1:
            selected_voice = voices[1].id
        else:
            selected_voice = voices[0].id

        engine.setProperty('voice', selected_voice)

        # Save the speech output to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_filename = fp.name

        engine.save_to_file(text, temp_filename)
        engine.runAndWait()

        # Read the generated audio and encode it as base64
        with open(temp_filename, "rb") as f:
            audio_data = f.read()
        os.remove(temp_filename)
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def play_audio(text):
    """Auto-play audio by embedding the generated audio in HTML."""
    audio_base64 = text_to_speech(text)
    if audio_base64:
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
        """
        st.components.v1.html(audio_html, height=0)

def render_ui():
    """Render the main user interface components."""
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>
            🍳 AI Cooking Assistant (Voice)
        </h1>
    """, unsafe_allow_html=True)
    st.caption("Use the microphone to ask questions about recipes!")

def handle_file_processing(uploaded_file):
    """Handle document processing workflow using DocProcessor."""
    try:
        # if os.path.exists(PERSIST_DIRECTORY):
        #     shutil.rmtree(PERSIST_DIRECTORY)
        
        with st.spinner("Processing your recipe..."):
            doc_processor = DocProcessor(EMBEDDING_MODEL)
            documents = doc_processor.load_and_split(uploaded_file)
            st.session_state.vector_retriever = doc_processor.create_vector_store(documents)
            st.session_state.current_document = uploaded_file.name
            st.success(f"Loaded {uploaded_file.name}! Ask me anything about this recipe.")
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.session_state.current_document = None

def handle_user_query(question):
    """Process and respond to a user query using ResponseGenerator."""
    with st.spinner("wait..."):
        try:
            response_gen = ResponseGenerator(
                retriever=st.session_state.vector_retriever,
                history=st.session_state.conversation_history
            )
            response = response_gen.generate(question)
            # Ensure response is a string
            if isinstance(response, dict) and 'content' in response:
                response = response['content']
            elif not isinstance(response, str):
                response = str(response)
            st.session_state.conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ])
            # Display and speak response
            with st.chat_message("assistant"):
                st.write(response)
                play_audio(response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

def continuous_conversation():
    """
    Continuously listen and respond while the conversation is active.
    After processing one query, the app reruns to listen again.
    """
    question = recognize_speech()
    if question:
        with st.chat_message("user"):
            st.write(question)
        if st.session_state.vector_retriever:
            handle_user_query(question)
        else:
            try:
                doc_processor = DocProcessor(EMBEDDING_MODEL)
                st.session_state.vector_retriever = doc_processor.create_vector_store()
                st.session_state.current_document = "Existing Knowledge Base"
                handle_user_query(question)
            except Exception as e:
                st.error("No recipe loaded and no existing knowledge found. Please upload a PDF first.")
    # Rerun the app to continue the loop if conversation is still active.
    st.rerun()

def main():
    initialize_session_state()
    render_ui()
    
    # Let the user choose the voice for output before starting the conversation.
    voice_choice = st.selectbox("Select Voice for Output", ["Girl", "Boy"])
    st.session_state.selected_voice = voice_choice

    # Sidebar for document management (without conversation controls)
    with st.sidebar:
        st.header("Document Management")
        uploaded_file = st.file_uploader("Upload Recipe PDF", type="pdf")
        if st.button("Process New Recipe"):
            if uploaded_file:
                handle_file_processing(uploaded_file)
            else:
                st.warning("Please upload a PDF file first")
        if st.session_state.current_document:
            st.divider()
            st.subheader("Current Recipe")
            st.write(f"📄 {st.session_state.current_document}")
    
    # Create a single toggle button on the right side using columns
    col1, col2 = st.columns([3, 1])
    with col2:
        toggle_label = "Stop Conversation" if st.session_state.conversation_active else "Start Conversation"
        if st.button(toggle_label, key="toggle_conv"):
            st.session_state.conversation_active = not st.session_state.conversation_active
    
    # Display conversation history
    with st.container():
        for entry in st.session_state.conversation_history:
            with st.chat_message(entry["role"]):
                st.write(entry["content"])
    
    # If conversation is active, continuously listen and respond.
    if st.session_state.conversation_active:
        continuous_conversation()

if __name__ == "__main__":
    main()
