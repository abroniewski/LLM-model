import streamlit as st
import requests
import datetime
import sys

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"  # Ollama's OpenAI-compatible endpoint
DEFAULT_MODEL = "phi"  # Default model name
MODEL_CONTEXT_LIMIT = 2048  # Max tokens for model context
SUMMARIZE_THRESHOLD = 1500  # When to trigger summarization
MAX_RESPONSE_TOKENS = 1024  # Lowered for faster responses

# --- System Prompts ---
SYSTEM_PROMPT_GENERAL = (
    "You are a concise, helpful assistant. "
    "Only answer the user's latest question. "
    "If you see previous conversation history, use it for context, but do not repeat or summarize it. "
    "Always respond as if you are in a chat, and do not generate puzzles, games, or extra content unless explicitly asked."
    "Your primary goal is to generate working code for the app or idea the user wants to build. Generate the code and be specific about which files the code belongs to. Include succinct descriptions that explain the purpose of the code and how it works."
)
SYSTEM_PROMPT_PHI = (
    "You are a helpful assistant. Answer the user's question directly and concisely."
)

# --- Model Types ---
SINGLE_TURN_MODELS = {"phi"}  # Models that only support single-turn Q&A

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Local Ollama Chat", layout="centered")
st.title("Local LLM Chat")

# --- Session State Initialization ---
# Store chat history, summary, and logs in session state for persistence across reruns
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Each item: {role: 'user'|'assistant'|'system', content: str}
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- Logging Function ---
def log(msg, level="INFO"):
    """Log messages to both the sidebar and the terminal, with color coding."""
    st.session_state.logs.append(f"[{level}] {msg}")
    now = datetime.datetime.now().strftime('%H:%M:%S')
    color = {
        "INFO": "\033[94m",   # Blue
        "ERROR": "\033[91m",  # Red
        "WARN": "\033[93m",   # Yellow
        "DEBUG": "\033[92m",  # Green
    }.get(level, "\033[0m")
    reset = "\033[0m"
    print(f"{color}[{now}] [{level}] {msg}{reset}", file=sys.stderr if level=="ERROR" else sys.stdout)

# --- Sidebar: Model Selection and Debug Log ---
with st.sidebar:
    # List of available models (edit this list to add more)
    available_models = [
        "phi:latest",                # Smallest, fastest
        "codellama:instruct",        # Small, general-purpose
        "codellama:7b-instruct",     # Small, chat-tuned
        "deepseek-coder:6.7b",       # Small, code-focused
        "codellama:13b-instruct",    # Mid-range, balanced
        "codellama:34b-instruct"     # Largest, slowest, highest quality
    ]
    default_model = "codellama:7b-instruct" if "codellama:7b-instruct" in available_models else available_models[0]
    model = st.selectbox("Model name", available_models, index=available_models.index(default_model))
    st.markdown("""
    - Make sure Ollama is running locally.
    - Only available models are shown here.
    - This app uses the OpenAI-compatible /v1/chat/completions endpoint.
    - For best chat results, use a chat/instruct model like 'codellama:7b-instruct' or 'deepseek-coder:6.7b'.
    """)
    # Show model-specific notes
    if model.startswith("phi"):
        st.info("\n**Note:** The selected model ('phi') is best for single-turn Q&A, short completions, and direct prompts. It does not have memory or handle multi-turn conversation well. For chat, use a model like 'codellama:7b-instruct' or 'deepseek-coder:6.7b'.\n")
    st.markdown("---")
    st.markdown("#### Debug Log")
    # Show the last 20 log entries
    for entry in st.session_state.logs[-20:]:
        st.text(entry)

# --- Main Chat History Display ---
with st.container():
    st.markdown("#### Chat History")
    st.markdown(
        """
        <div style='max-height: 400px; overflow-y: auto; border: 1px solid #444; border-radius: 8px; padding: 1em; background: #18191A;'>
        """,
        unsafe_allow_html=True
    )
    # Render each message in the chat history
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg['role'] not in ('user', 'assistant'):
            continue
        content = msg['content'].strip()
        if not content:
            continue  # skip empty messages
        is_user = msg['role'] == 'user'
        align = 'right' if is_user else 'left'
        margin = 'margin-left: 25%;' if is_user else 'margin-right: 25%;'
        box_style = (
            f"background: #23272F; border: 1px solid #444; border-radius: 8px; padding: 0.7em 1em; margin: 0.5em 0; max-width: 75%; {margin} text-align: {align};"
        )
        st.markdown(f"<div style='{box_style}'>", unsafe_allow_html=True)
        # Render as code if it looks like code, otherwise markdown
        if '```' in content:
            import re
            parts = re.split(r'(```[a-zA-Z]*\n[\s\S]*?```)', content)
            for part in parts:
                if part.startswith('```'):
                    code = part.strip('`').split('\n', 1)
                    lang = code[0][3:].strip() if len(code[0]) > 3 else ''
                    code_content = code[1] if len(code) > 1 else ''
                    st.code(code_content, language=lang)
                elif part.strip():
                    st.markdown(part, unsafe_allow_html=False)
        elif content.startswith(('def ', 'class ', 'import ', 'for ', 'while ')):
            st.code(content, language='python')
        else:
            st.markdown(content, unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input Form ---
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_area("Enter your prompt:", height=150, key="prompt_input")
    submitted = st.form_submit_button("Send")

# --- Keyboard Shortcut for Submission ---
st.markdown(
    """
    <script>
    const textarea = window.parent.document.querySelector('textarea[data-testid=\"stTextArea\"]');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                const btn = window.parent.document.querySelector('button[kind=\"primary\"]');
                if (btn) btn.click();
            }
        });
    }
    </script>
    """,
    unsafe_allow_html=True
)

# --- Prevent Double Submission on Rerun ---
if 'handled_submit' not in st.session_state:
    st.session_state.handled_submit = False

# --- Token Estimation Helper ---
def estimate_tokens(text):
    """Roughly estimate the number of tokens in a string (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)

# --- Message Builders ---
def build_single_turn_messages(user_prompt):
    """For models like phi: only system prompt and latest user message."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PHI},
        {"role": "user", "content": user_prompt}
    ]

def build_multi_turn_messages(user_prompt):
    """For chat/instruct models: system prompt, summary, last 4 messages, separator, new user prompt."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT_GENERAL}]
    if st.session_state.summary:
        messages.append({"role": "system", "content": f"[Summary so far]: {st.session_state.summary}"})
    history = st.session_state.chat_history[-4:]
    if history:
        messages.extend(history)
    messages.append({"role": "system", "content": "--- End of previous chat history. The next message is a new user prompt. ---"})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def build_messages(user_prompt, model_name):
    """Dispatch to the correct message builder based on model."""
    if model_name.strip().lower() in SINGLE_TURN_MODELS:
        return build_single_turn_messages(user_prompt)
    else:
        return build_multi_turn_messages(user_prompt)

# --- Summarization Logic ---
def maybe_summarize(model_name):
    """If the chat history is too long, summarize it to fit within the model's context window."""
    if model_name.strip().lower() in SINGLE_TURN_MODELS:
        return  # No summarization for single-turn models
    messages = build_multi_turn_messages("")  # No new user prompt for summary
    context_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    if estimate_tokens(context_text) > SUMMARIZE_THRESHOLD:
        summary_prompt = [
            {"role": "system", "content": "Summarize the following conversation in a concise way, preserving important details and context for future questions."},
        ] + messages
        payload = {
            "model": model_name,
            "messages": summary_prompt,
            "stream": False,
            "max_tokens": 128
        }
        log(f"[SUMMARIZE] Payload: {payload}")
        print(f"[SUMMARIZE] Sending payload: {payload}")
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=240)
            print(f"[SUMMARIZE] Raw response: {response.text}")
            log(f"[SUMMARIZE] Raw response: {response.text}")
            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                summary = data["choices"][0]["message"]["content"]
                st.session_state.summary = summary.strip()
                st.session_state.chat_history = st.session_state.chat_history[-4:]
                log(f"[SUMMARIZE] Summary: {summary.strip()}")
                print(f"[SUMMARIZE] Summary: {summary.strip()}")
                st.info("Chat history summarized to fit model context window.")
            else:
                log(f"[SUMMARIZE] Unexpected response: {data}")
                print(f"[SUMMARIZE] Unexpected response: {data}")
                st.warning("No summary received from Ollama.")
        except Exception as e:
            log(f"[SUMMARIZE] Error: {e}", level="ERROR")
            print(f"[SUMMARIZE] Error: {e}")
            st.warning(f"Failed to summarize conversation: {e}")

# --- Main Chat Submission Logic ---
if submitted and prompt.strip() and not st.session_state.handled_submit:
    st.session_state.handled_submit = True
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt.strip()})
    messages = build_messages(prompt.strip(), model)
    # Only summarize for multi-turn models
    if model.strip().lower() not in SINGLE_TURN_MODELS:
        context_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        if estimate_tokens(context_text) > SUMMARIZE_THRESHOLD:
            maybe_summarize(model)
            messages = build_messages(prompt.strip(), model)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # Disable streaming
        "max_tokens": MAX_RESPONSE_TOKENS
    }
    log(f"[SEND] Payload: {payload}")
    print(f"[SEND] Sending payload: {payload}")
    with st.spinner("Thinking..."):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            if not response.text.strip():
                st.error("No response received from Ollama. The model may have failed to generate output or crashed.")
                log("No response received from Ollama.", level="ERROR")
            else:
                data = response.json()
                if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                    answer = data["choices"][0]["message"]["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": answer.strip()})
                    log(f"[SEND] Response: {answer.strip()}")
                    print(f"[SEND] Response: {answer.strip()}")
                else:
                    log(f"[SEND] Unexpected response: {data}")
                    print(f"[SEND] Unexpected response: {data}")
                    st.warning("No response received from Ollama.")
            try:
                st.rerun()
            except Exception as rerun_error:
                log(f"[RERUN] Error: {rerun_error}", level="ERROR")
                print(f"[RERUN] Error: {rerun_error}")
        except Exception as e:
            st.error(f"Error: {e}")
            log(f"[SEND] Error: {e}", level="ERROR")
            print(f"[SEND] Error: {e}")

# --- Reset handled_submit flag after rerun ---
if not submitted:
    st.session_state.handled_submit = False 