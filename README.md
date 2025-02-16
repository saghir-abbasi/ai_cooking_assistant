# AI Cooking Assistant

## Overview
The **AI Cooking Assistant** is a real-time, voice-to-voice, RAG-based application that helps users cook recipes interactively. The assistant processes PDF recipe documents and provides step-by-step cooking guidance while maintaining conversational context. Users can select a chef or a specific recipe, and the AI will assist with instructions dynamically.

## Features
- **Voice-to-Voice Interaction:** Converts speech to text and provides AI-generated responses using text-to-speech.
- **Recipe Guidance:** Processes PDF documents and retrieves relevant recipe steps.
- **Conversational Context:** Maintains context during the cooking session.
- **Multi-Chef Support:** Users can select different chefs for customized cooking instructions.
- **Powered by RAG:** Uses Retrieval-Augmented Generation (RAG) to fetch accurate recipe information.

## Technologies Used
- **Streamlit** – Frontend framework for UI.
- **SpeechRecognition** – Converts user voice input to text.
- **pyttsx3** – Text-to-speech conversion.
- **LangChain** – Used for document processing and AI interaction.
- **Google Generative AI** – Language model for response generation.
- **Pinecone** – Vector database for efficient recipe retrieval.
- **Python-Dotenv** – Manages environment variables.

## Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/saghir-abbasi/ai_cooking_assistant.git
   cd ai-cooking-assistant
   ```
2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```


## Usage
Run the Streamlit app with:
```sh
streamlit run app.py
```

## How It Works
1. The user selects a recipe PDF.
2. The AI assistant extracts and vectorizes the content.
3. The user can interact with the AI via voice commands.
4. The AI provides step-by-step cooking instructions.

## Future Enhancements
- Support for video-based cooking tutorials.
- Integration with smart kitchen appliances.
- Multi-language support.

## Contributing
Feel free to submit issues or pull requests to improve this project.

## License
This project is licensed under the MIT License.

