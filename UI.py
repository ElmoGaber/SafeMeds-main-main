# ui.py
import gradio as gr
from backend import gemini_chat_wrapper   # Import the function from backend.py

# -------------------------------
# Gradio Chat Interface
# -------------------------------
# Gradio provides a quick way to create a web-based UI for ML/AI apps.
# Here, we use ChatInterface which automatically builds a chatbot UI.

demo = gr.ChatInterface(
    fn=gemini_chat_wrapper,  # The backend function that handles user queries
    title="SafeMeds RAG Chatbot 🤖",  # Title shown in the UI
    description="Ask me anything from our information. I'm powered by SafeMeds RAG Chatbot 🤖",  # Shown below the title
)

# -------------------------------
# Run the Gradio app
# -------------------------------
# share=True -> generates a public link via Gradio’s servers (useful for testing/demo)
if __name__ == "__main__":
    demo.launch(share=True)
