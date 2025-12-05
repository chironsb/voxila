"""
Main entry point for Speech-to-RAG v2
Orchestrates all components for a complete conversation loop
"""
import os
import sys
import uuid
from dotenv import load_dotenv

# Ensure UTF-8 encoding for terminal output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress PyTorch deprecation warnings for stft
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*stft with return_complex=False.*')

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from audio import AudioRecorder, AudioPlayer
from asr import ASR
from tts import TTS
from llm import LLM
from rag import RAG

# Try to import TTS modules
try:
    from tts_openvoice import OpenVoiceTTS
    OPENVOICE_AVAILABLE = True
except ImportError:
    OPENVOICE_AVAILABLE = False

try:
    from tts_coqui import CoquiTTSWrapper
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

# ANSI escape codes for colors
CYAN = '\033[96m'
PINK = '\033[95m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Load environment variables
load_dotenv()


def load_system_prompt(prompt_file: str = "chatbot2.txt") -> str:
    """Load system prompt from file or use default"""
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        # Default system prompt
        return """You are a helpful AI assistant. You answer questions based on the context provided.
Be concise, accurate, and friendly in your responses."""


class SpeechToRAG:
    """Main Speech-to-RAG application"""
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing Speech-to-RAG v2...")
        
        # Load configuration
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        show_reasoning = os.getenv("SHOW_REASONING", "false").lower() == "true"
        disable_reasoning = os.getenv("DISABLE_REASONING", "false").lower() == "true"
        whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
        whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        piper_binary = os.getenv("PIPER_BINARY_PATH", "./models/piper/piper")
        piper_model = os.getenv("PIPER_MODEL_PATH", "./models/piper/en_US-amy-medium.onnx")
        openvoice_checkpoints = os.getenv("OPENVOICE_CHECKPOINTS", "./checkpoints")
        openvoice_reference = os.getenv("OPENVOICE_REFERENCE", "./voxila.mp3")
        coqui_tts_model = os.getenv("COQUI_TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        vault_file = os.getenv("VAULT_FILE", "./data/vault.txt")
        faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./data/embeddings/faiss_index")
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
        
        # Setup local temp directory for audio files
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(src_dir)
        self.temp_dir = os.path.join(project_root, 'tmp')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize components
        print("Loading components...")
        self.audio_recorder = AudioRecorder()
        self.audio_player = AudioPlayer()
        self.asr = ASR(model_size=whisper_model_size, device=whisper_device)
        
        # TTS - try OpenVoice first, then Coqui TTS, then Piper
        self.tts = None
        self.tts_type = None
        
        # Try OpenVoice (like original project)
        if OPENVOICE_AVAILABLE:
            try:
                device = os.getenv("OPENVOICE_DEVICE", "cpu")  # Default to CPU to avoid GPU memory issues
                if device == "auto":
                    device = None  # Let OpenVoice auto-detect (will default to CPU)
                self.tts = OpenVoiceTTS(
                    checkpoints_dir=openvoice_checkpoints,
                    reference_voice=openvoice_reference,
                    device=device
                )
                self.tts_type = "openvoice"
                print("Using OpenVoice TTS (voice cloning enabled)")
            except (FileNotFoundError, Exception) as e:
                print(f"OpenVoice TTS not available: {e}")
                if "checkpoints" in str(e).lower():
                    print("Tip: Download checkpoints from: https://nordnet.blob.core.windows.net/bilde/checkpoints.zip")
                print("Trying Coqui TTS...")
        
        # Fallback to Coqui TTS (simpler, no checkpoints needed)
        # Note: Coqui TTS requires Python < 3.13, so it may not work on Python 3.13+
        if self.tts is None and COQUI_TTS_AVAILABLE:
            try:
                self.tts = CoquiTTSWrapper(model_name=coqui_tts_model)
                self.tts_type = "coqui"
                print("Using Coqui TTS")
            except Exception as e:
                print(f"Coqui TTS not available: {e}")
                print("Note: Coqui TTS requires Python < 3.13")
                print("Trying Piper TTS...")
        
        # Fallback to Piper TTS
        if self.tts is None:
            try:
                self.tts = TTS(piper_binary, piper_model)
                self.tts_type = "piper"
                print("Using Piper TTS")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                print("TTS will be disabled. Please install TTS dependencies.")
                self.tts = None
        
        self.llm = LLM(model=ollama_model, base_url=ollama_base_url, show_reasoning=show_reasoning, disable_reasoning=disable_reasoning)
        self.show_reasoning = show_reasoning
        self.disable_reasoning = disable_reasoning
        
        self.rag = RAG(
            embedding_model=embedding_model,
            vault_file=vault_file,
            faiss_index_path=faiss_index_path
        )
        
        # Load system prompt
        self.system_prompt = load_system_prompt()
        # Add instruction to disable reasoning if requested
        if self.disable_reasoning:
            self.system_prompt += "\n\nCRITICAL: You MUST respond directly without any thinking tags, reasoning tags, or internal monologue. Do NOT use <think>, </think>, <reasoning>, </reasoning>, or any similar tags. Give your answer immediately and directly. No thinking process, no analysis, just the answer."
        self.llm.set_system_prompt(self.system_prompt)
        
        print("Initialization complete!\n")
    
    def handle_special_commands(self, user_input: str) -> bool:
        """
        Handle special commands. Returns True if command was handled.
        
        Args:
            user_input: User input text
            
        Returns:
            True if command was handled, False otherwise
        """
        user_input_lower = user_input.lower().strip()
        
        # Exit command
        if user_input_lower == "exit":
            print(NEON_GREEN + "Goodbye!" + RESET_COLOR)
            return True
        
        # Print info command
        if user_input_lower.startswith("print info") or user_input_lower.startswith("list info") or user_input_lower == "list":
            print(NEON_GREEN + "Info contents:" + RESET_COLOR)
            if os.path.exists(self.rag.vault_file):
                with open(self.rag.vault_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        print(NEON_GREEN + content + RESET_COLOR)
                    else:
                        print("Info is empty.")
            else:
                print("Info is empty.")
            return True
        
        # Use RAG command - explicitly enable RAG for this query
        if user_input_lower.startswith("use rag") or user_input_lower.startswith("with context") or user_input_lower.startswith("/rag"):
            # Extract the actual query after the command
            query = user_input_lower.replace("use rag", "").replace("with context", "").replace("/rag", "").strip()
            if not query:
                print(YELLOW + "Usage: 'use rag <your question>' or '/rag <your question>'" + RESET_COLOR)
                return True
            
            # Get context from RAG
            context = None
            if (os.path.exists(self.rag.vault_file) and 
                os.path.getsize(self.rag.vault_file) > 0 and 
                len(self.rag.vault_content) > 0):
                context = self.rag.get_context(query, top_k=self.rag_top_k)
                print(CYAN + f"You (with RAG): {query}" + RESET_COLOR)
                print(PINK + "Assistant: " + RESET_COLOR, end="", flush=True)
                
                # Get LLM response with RAG context
                response = self.llm.chat(
                    query,
                    context=context if context else None,
                    stream=True,
                    stream_callback=None
                )
                
                if response and response.strip():
                    print(NEON_GREEN + response.strip() + RESET_COLOR)
                print()
                return True
            else:
                print(YELLOW + "RAG vault is empty. Use 'insert info' to add information first." + RESET_COLOR)
                return True
        
        # Delete info command
        if user_input_lower.startswith("delete info"):
            print(YELLOW + "Are you sure? Say 'yes' to confirm (or 'no' to cancel):" + RESET_COLOR)
            # Record confirmation with timeout (5 seconds max) or Enter to stop
            temp_file = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex[:8]}.wav")
            self.audio_recorder.record_to_file(temp_file, duration=5.0, stop_on_enter=True)
            
            try:
                confirmation = self.asr.transcribe(temp_file)
                os.remove(temp_file)
                
                # Clean and normalize confirmation
                confirmation_clean = confirmation.lower().strip()
                # Remove punctuation and extra spaces
                import re
                confirmation_clean = re.sub(r'[^\w\s]', '', confirmation_clean)
                confirmation_clean = re.sub(r'\s+', ' ', confirmation_clean).strip()
                
                # Check if confirmation is "yes" (more flexible matching)
                # Accept: "yes", "yes please", "i said yes", "yeah", "yep", etc.
                confirmation_words = confirmation_clean.split()
                is_yes = (
                    confirmation_clean == "yes" or 
                    confirmation_clean.startswith("yes") or 
                    "yes" in confirmation_words or
                    confirmation_clean == "yeah" or
                    confirmation_clean == "yep" or
                    confirmation_clean == "y" or
                    confirmation_clean.startswith("yeah") or
                    confirmation_clean.startswith("yep")
                )
                
                if is_yes:
                    self.rag.remove_all_documents()
                    if os.path.exists(self.rag.vault_file):
                        os.remove(self.rag.vault_file)
                    print(NEON_GREEN + "Info deleted." + RESET_COLOR)
                else:
                    print(f"Info deletion cancelled. (You said: '{confirmation}')")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return True
        
        # Insert info command (also accept variations)
        if user_input_lower.startswith("insert info") or user_input_lower.startswith("inserting info") or "insert info" in user_input_lower:
            print(YELLOW + "Recording for info... (Press ENTER to stop)" + RESET_COLOR)
            temp_file = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex[:8]}.wav")
            self.audio_recorder.record_to_file(temp_file, stop_on_enter=True)
            
            try:
                vault_input = self.asr.transcribe(temp_file)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if vault_input.strip():
                    # Ensure vault file directory exists
                    vault_dir = os.path.dirname(self.rag.vault_file)
                    if vault_dir and not os.path.exists(vault_dir):
                        os.makedirs(vault_dir, exist_ok=True)
                    
                    # Add to vault file (create if doesn't exist)
                    with open(self.rag.vault_file, 'a', encoding='utf-8') as f:
                        f.write(vault_input + "\n")
                    
                    # Add to RAG index
                    self.rag.add_document(vault_input)
                    print(NEON_GREEN + f"Wrote to info: {vault_input}" + RESET_COLOR)
                else:
                    print("No content recorded.")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return True
        
        return False
    
    def conversation_loop(self):
        """Main conversation loop with text and voice input support"""
        print(CYAN + "=" * 60 + RESET_COLOR)
        print(NEON_GREEN + "Speech-to-RAG v2 - Ready!" + RESET_COLOR)
        print(CYAN + "=" * 60 + RESET_COLOR)
        print("Usage:")
        print("  - Type your message and press ENTER for text input")
        print("  - Press ENTER (empty) to start voice recording")
        print("  - Press ENTER again during recording to stop")
        print("  - Commands:")
        print("    '/mode text'  - Text output mode (responses are text only)")
        print("    '/mode voice' - Voice output mode (responses are spoken)")
        print("    'q' or '/quit' - Quit")
        print("\nVoice Commands (when using voice input):")
        print("  - Say 'exit' to quit")
        print("  - Say 'print info' to see knowledge base")
        print("  - Say 'delete info' to clear knowledge base")
        print("  - Say 'insert info' to add information")
        print(CYAN + "=" * 60 + RESET_COLOR + "\n")
        
        # Settings
        mode = 'voice'  # 'text' or 'voice' - determines output format (text or TTS)
        
        while True:
            temp_audio = None
            try:
                # Get user input
                prompt = YELLOW + "You (type message or press ENTER for voice): " + RESET_COLOR
                user_input_raw = input(prompt).strip()
                
                # Handle quit
                if user_input_raw.lower() in ['q', '/quit', 'quit']:
                    print(NEON_GREEN + "Goodbye!" + RESET_COLOR)
                    break
                
                # Handle commands
                if user_input_raw.startswith('/'):
                    if user_input_raw.lower() == '/mode text':
                        mode = 'text'
                        print(NEON_GREEN + "Mode set to text output (responses will be text only)." + RESET_COLOR)
                        continue
                    elif user_input_raw.lower() == '/mode voice':
                        mode = 'voice'
                        print(NEON_GREEN + "Mode set to voice output (responses will be spoken)." + RESET_COLOR)
                        continue
                    else:
                        print(YELLOW + f"Unknown command: {user_input_raw}. Commands: '/mode text', '/mode voice', 'q' to quit." + RESET_COLOR)
                        continue
                
                # Determine input mode
                if user_input_raw == '':
                    # Empty input -> voice mode
                    input_mode = 'voice'
                else:
                    # Text input
                    input_mode = 'text'
                    user_input = user_input_raw
                
                # Voice input
                if input_mode == 'voice':
                    temp_audio = os.path.join(self.temp_dir, f"temp_audio_{uuid.uuid4().hex[:8]}.wav")
                    self.audio_recorder.record_to_file(temp_audio, stop_on_enter=True)
                    
                    # Transcribe
                    if temp_audio and os.path.exists(temp_audio):
                        user_input = self.asr.transcribe(temp_audio)
                        os.remove(temp_audio)
                        temp_audio = None
                    else:
                        user_input = ""
                    
                    if not user_input.strip():
                        print(YELLOW + "No speech detected. Try again." + RESET_COLOR)
                        continue
                
                # Handle special commands
                if self.handle_special_commands(user_input):
                    continue
                
                # Display user input
                print(CYAN + f"You: {user_input}" + RESET_COLOR)
                
                # RAG is disabled by default - only use when explicitly requested
                # For normal conversations, don't use RAG context
                context = None
                
                # Get LLM response
                print(PINK + "Assistant: " + RESET_COLOR, end="", flush=True)
                
                # Show streaming only if show_reasoning is enabled
                # Otherwise, we'll show the filtered response at the end
                def stream_callback(chunk: str):
                    """Callback for streaming response"""
                    if self.show_reasoning:
                        print(NEON_GREEN + chunk + RESET_COLOR, end="", flush=True)
                
                try:
                    response = self.llm.chat(
                        user_input,
                        context=context if context else None,
                        stream=True,
                        stream_callback=stream_callback if self.show_reasoning else None
                    )
                    if self.show_reasoning:
                        print()  # New line after streaming
                except KeyboardInterrupt:
                    # If user interrupts during LLM generation, just continue
                    print("\n" + YELLOW + "Response generation cancelled." + RESET_COLOR)
                    continue
                
                # Display the filtered response (reasoning already removed by llm._filter_reasoning)
                if response and response.strip():
                    # Only print if we didn't already stream it
                    if not self.show_reasoning:
                        print(NEON_GREEN + response.strip() + RESET_COLOR)
                else:
                    print(YELLOW + "(Response was empty after filtering)" + RESET_COLOR)
                
                # Limit conversation history (do this less frequently to avoid overhead)
                if len(self.llm.conversation_history) > 25:
                    self.llm.limit_history(max_messages=20)
                
                # Convert response to speech (if mode is voice)
                if mode == 'voice' and self.tts:
                    print(YELLOW + "Generating speech..." + RESET_COLOR)
                    try:
                        # Extract only the final response (after reasoning tags) for TTS
                        # Even if show_reasoning is True, we only want the final answer for TTS
                        import re
                        
                        # Get the raw response before any filtering
                        raw_response = response
                        
                        # Find the closing reasoning tag and extract ONLY what's after it
                        # Try various tag patterns - Qwen3 uses </think> (with backslash)
                        tts_text = ""
                        patterns = [r'</\\think>', r'</think>', r'</think>', r'</thinking>', r'</reasoning>', r'</think>']
                        tag_found = False
                        
                        for pattern in patterns:
                            match = re.search(pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
                            if match:
                                # Get everything AFTER the closing tag
                                tts_text = raw_response[match.end():].strip()
                                tag_found = True
                                break
                        
                        # If no tag found, try to find the last line that looks like a real response
                        if not tag_found or not tts_text:
                            lines = raw_response.split('\n')
                            reasoning_keywords = ['Okay,', 'First,', 'Wait,', 'Let me', 'I should', 'Maybe', 'Alternatively', 'So,', 'Hmm,', 'Yes,', 'Final', 'I think', 'I need', 'Since', 'I\'ll', 'I\'m', 'The user', 'In my response', 'Wait, the user']
                            
                            # Look for the last line that's not reasoning
                            for line in reversed(lines):
                                line = line.strip()
                                if not line or len(line) < 10:
                                    continue
                                # Skip reasoning lines
                                if any(line.startswith(kw) for kw in reasoning_keywords):
                                    continue
                                # Skip IPA lines
                                if re.match(r'^[aɪəɹɛæɔʊθðʃʒŋɑɪˈˌ\s]+$', line):
                                    continue
                                # Skip lines with "think" in them (likely reasoning)
                                if 'think' in line.lower() and len(line) < 50:
                                    continue
                                # This looks like a real response
                                tts_text = line
                                break
                        
                        # Final cleanup - remove any remaining reasoning patterns
                        if tts_text:
                            # Remove lines that start with reasoning keywords
                            lines = tts_text.split('\n')
                            filtered_lines = []
                            reasoning_keywords = ['Okay,', 'First,', 'Wait,', 'Let me', 'I should', 'Maybe', 'Alternatively', 'So,', 'Hmm,', 'Yes,', 'Final', 'I think', 'I need', 'Since', 'I\'ll', 'I\'m', 'The user', 'In my response', 'Wait, the user']
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                # Skip reasoning lines
                                if any(line.startswith(kw) for kw in reasoning_keywords):
                                    continue
                                # Skip lines with "think" (likely reasoning)
                                if 'think' in line.lower() and ('I think' in line or 'thinking' in line.lower()):
                                    continue
                                # Skip IPA lines
                                if re.match(r'^[aɪəɹɛæɔʊθðʃʒŋɑɪˈˌ\s]+$', line):
                                    continue
                                # Skip lines that are mostly IPA
                                if len(re.findall(r'[aɪəɹɛæɔʊθðʃʒŋɑɪˈˌ]', line)) > len(line) * 0.5:
                                    continue
                                filtered_lines.append(line)
                            
                            # Join filtered lines
                            tts_text = ' '.join(filtered_lines).strip()
                        
                        # Final cleanup
                        tts_text = re.sub(r'\s+', ' ', tts_text).strip()
                        # Remove emojis but keep text
                        tts_text = re.sub(r'[😊😄😃😁]', '', tts_text).strip()
                        # Remove any remaining "think" words if they're standalone
                        tts_text = re.sub(r'\bthink\b', '', tts_text, flags=re.IGNORECASE).strip()
                        tts_text = re.sub(r'\s+', ' ', tts_text).strip()
                        
                        # Use the filtered response directly if no tag was found
                        if not tts_text:
                            tts_text = response.strip()
                            # Clean it up
                            tts_text = re.sub(r'[😊😄😃😁]', '', tts_text).strip()
                            tts_text = re.sub(r'\s+', ' ', tts_text).strip()
                        
                        if tts_text and len(tts_text) >= 3:  # Minimum 3 characters for TTS
                            try:
                                if self.tts_type == "openvoice":
                                    self.tts.synthesize_and_play(tts_text)
                                elif self.tts_type == "coqui":
                                    self.tts.synthesize_and_play(tts_text)
                                elif self.tts_type == "piper":
                                    temp_tts = os.path.join(self.temp_dir, f"temp_tts_{uuid.uuid4().hex[:8]}.wav")
                                    self.tts.synthesize(tts_text, temp_tts)
                                    self.audio_player.play_file(temp_tts)
                                    if os.path.exists(temp_tts):
                                        os.remove(temp_tts)
                            except Exception as tts_error:
                                print(f"Error generating speech: {tts_error}")
                                import traceback
                                traceback.print_exc()
                    except Exception as e:
                        print(f"Error generating speech: {e}")
                        import traceback
                        traceback.print_exc()
                
                print()  # Blank line between turns
                
            except KeyboardInterrupt:
                # Handle Ctrl+C as quit signal
                print("\n" + YELLOW + "Interrupted. Use 'q' to quit." + RESET_COLOR)
                continue
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def cleanup(self):
        """Clean up resources"""
        self.audio_recorder.cleanup()
        self.audio_player.cleanup()


def main():
    """Main entry point"""
    app = SpeechToRAG()
    try:
        app.conversation_loop()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()

