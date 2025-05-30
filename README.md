# Local Ollama LLM Chat UI

A simple, instructional web UI to chat with your Ollama LLM models using Streamlit. This project is designed for learning, experimentation, and local privacy. It demonstrates best practices for building your own local LLM chat interface, with clear code structure and comments to help you adapt it for your own needs.

---

## Features
- **Chat with local LLMs**: Interact with any model you have pulled in Ollama, using a familiar chat interface.
- **Model selection**: Choose from a range of models in the sidebar, ordered from fastest/simplest to slowest/most capable.
- **Performance tuning**: Easily adjust max response tokens, summarization thresholds, and model selection for your hardware and needs.
- **Context-aware chat**: For chat/instruct models, the app maintains conversation history and summarizes it when it gets too long.
- **Single-turn mode**: For models like `phi`, the app only sends your latest prompt (no chat history).
- **Debug log**: See request/response logs in the sidebar for transparency and troubleshooting.
- **Keyboard shortcut**: Submit your prompt with Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux).
- **Code generation focus**: The system prompt is tuned to generate working code for user ideas, specifying file names and providing succinct explanations.

---

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- At least one model pulled (e.g., `ollama pull codellama:7b-instruct`)
- `streamlit` and `requests` Python packages (see below)

---

## Setup
```bash
pip install -r requirements.txt
```

---

## Running the App
```bash
streamlit run app.py
```

Open the link shown in your terminal (usually http://localhost:8501) in your browser.

---

## Usage Instructions
1. **Start Ollama**: Make sure Ollama is running locally (`ollama serve`).
2. **Select a Model**: Use the sidebar to pick from your available models. Models are ordered from fastest/simplest to slowest/most capable:
    - `phi:latest` (smallest, fastest)
    - `codellama:instruct` (small, general-purpose)
    - `codellama:7b-instruct` (small, chat-tuned)
    - `deepseek-coder:6.7b` (small, code-focused)
    - `codellama:13b-instruct` (mid-range, balanced)
    - `codellama:34b-instruct` (largest, slowest, highest quality)
3. **Enter a Prompt**: Type your question or instruction in the text area and click "Send" (or press Cmd/Ctrl+Enter).
4. **View Responses**: The assistant's reply will appear in the chat history. Code blocks and markdown are rendered for readability.
5. **Debug Log**: Check the sidebar for request/response logs and troubleshooting info.

---

## How the App Works (Logic Overview)

- **Session State**: The app uses Streamlit's `st.session_state` to store chat history, summaries, and logs. This keeps your conversation persistent across reruns.
- **Model Selection & Ordering**: The sidebar lists models from fastest to slowest, helping you choose the right balance of speed and quality for your needs. You can add or remove models in the `available_models` list in `app.py`.
- **Prompt Building**: The app builds the message payload according to the model type:
    - For chat/instruct models (like `codellama:7b-instruct`), it sends the last few messages and a summary if the conversation is long.
    - For single-turn models (like `phi`), it only sends your latest prompt (no history).
- **System Prompts**: The system prompt is designed to:
    - Encourage concise, direct answers
    - Focus on generating working code for user ideas, specifying file names and providing explanations
    - Avoid unnecessary repetition or extra content
- **Summarization**: If the chat history gets too long (over a token threshold), the app asks the model to summarize the conversation, then keeps only the summary and the last few messages. This helps stay within the model's context window.
- **Sending Requests**: When you submit a prompt, the app sends a POST request to Ollama's `/v1/chat/completions` endpoint, waits for a response, and displays it in the chat. (Streaming can be enabled or disabled in the code.)
- **Rendering**: The chat history is rendered with markdown and code highlighting. The app tries to detect and format code blocks for readability.
- **Debugging**: All requests, responses, and errors are logged to both the sidebar and the terminal for transparency.

---

## Performance Tuning & Model Selection Strategy

- **Model Size vs. Speed**: Smaller models (like `phi:latest` or `7b` variants) are much faster and use less memory, but may be less accurate or creative. Larger models (like `13b` or `34b`) provide better quality but are slower and require more resources.
- **Mid-Range Models**: For most users, a 13B model (e.g., `codellama:13b-instruct`) offers the best balance of speed and quality on modern hardware.
- **Max Tokens**: You can adjust `MAX_RESPONSE_TOKENS` in `app.py` to control the length of responses. Higher values allow longer answers but may slow down response time.
- **Streaming**: Streaming can be enabled for faster perceived responses, but may be disabled for compatibility or stability. Adjust the `stream` parameter in the code as needed.
- **Summarization Threshold**: The `SUMMARIZE_THRESHOLD` controls when the app summarizes chat history to fit within the model's context window. Tune this for your use case.
- **Hardware Acceleration**: Ollama uses Apple Silicon's Metal backend for GPU acceleration when running natively (not in Docker). For best performance, run Ollama natively on Mac.

---

## Extending the App
- Add more models to the `available_models` list in `app.py`.
- Change the system prompts to experiment with different assistant behaviors or focus areas (e.g., code generation, creative writing).
- Adjust the summarization threshold or max response tokens for longer or shorter conversations.
- Enable or disable streaming for different user experiences.
- Add new UI elements or features using Streamlit's API.
- Use the code as a template for your own local LLM projects, adapting the logic to your needs.

---

## Troubleshooting
- **No models listed?** Make sure you have pulled at least one model with `ollama pull <modelname>`.
- **Ollama not running?** Start it with `ollama serve` in your terminal.
- **Errors in chat?** Check the sidebar debug log and your terminal for details.
- **Slow responses or crashes?** Try a smaller model or lower the max response tokens.

---

## Code Comments & Learning
The code in `app.py` is heavily commented to explain each section and function. Read through the code for inline explanations of the logic and design choices. This project is intended to be a learning resource—feel free to adapt, extend, and experiment with the code to build your own local LLM chat UI! 