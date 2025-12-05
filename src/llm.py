"""
LLM integration module using Ollama
"""
import ollama
from typing import List, Dict, Optional, Callable


class LLM:
    """Large Language Model using Ollama"""
    
    def __init__(self, model: str = "qwen3:1.7b", base_url: str = "http://localhost:11434", show_reasoning: bool = False, disable_reasoning: bool = False):
        """
        Initialize the LLM client
        
        Args:
            model: Ollama model name
            base_url: Ollama API base URL
            show_reasoning: If True, keep reasoning in responses
            disable_reasoning: If True, try to stop reasoning early and limit tokens
        """
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self.conversation_history: List[Dict[str, str]] = []
        self.show_reasoning = show_reasoning
        self.disable_reasoning = disable_reasoning
    
    def chat(self, message: str, system_prompt: Optional[str] = None, 
             context: Optional[str] = None, stream: bool = False,
             stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Send a chat message to the LLM
        
        Args:
            message: User message
            system_prompt: System prompt (optional, can be set once)
            context: Additional context to include (e.g., from RAG)
            stream: Whether to stream the response
            stream_callback: Callback function for streaming (receives chunks)
            
        Returns:
            Full response text
        """
        # Build messages efficiently
        # Use system prompt from conversation history if available (set once)
        messages = []
        
        # Add system prompt if provided (or use from history)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif self.conversation_history and self.conversation_history[0].get('role') == 'system':
            # System prompt already in history, don't duplicate
            pass
        
        # Add conversation history (skip system if we just added it)
        start_idx = 0
        if messages and messages[0].get('role') == 'system' and self.conversation_history and self.conversation_history[0].get('role') == 'system':
            start_idx = 1  # Skip system message in history
        messages.extend(self.conversation_history[start_idx:])
        
        # Add context if provided
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {message}"
        else:
            user_message = message
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get response
        if stream:
            full_response = ""
            
            # Prepare options for Ollama - no restrictions, let it generate naturally
            options = {}
            
            for chunk in self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=options if options else None
            ):
                content = chunk.get('message', {}).get('content', '')
                if content:
                    full_response += content
                    if stream_callback:
                        stream_callback(content)
            
            # Filter out reasoning tags if present (but keep the response as-is otherwise)
            # Only filter if we need to (not keeping reasoning)
            if not self.show_reasoning:
                full_response = self._filter_reasoning(full_response, keep_reasoning=False)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            return full_response
        else:
            # Prepare options for Ollama - no restrictions, let it generate naturally
            options = {}
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options if options else None
            )
            
            full_response = response['message']['content']
            
            # Filter out reasoning/internal thinking
            full_response = self._filter_reasoning(full_response, keep_reasoning=self.show_reasoning)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            return full_response
    
    def _filter_reasoning(self, text: str, keep_reasoning: bool = False) -> str:
        """
        Filter out reasoning/internal thinking from response - FAST VERSION
        Simply finds closing tags and extracts what's after them
        
        Args:
            text: Text to filter
            keep_reasoning: If True, keep reasoning but format it nicely
        """
        import re
        
        if keep_reasoning:
            # Just clean up formatting but keep reasoning
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            return text.strip()
        
        # Fast path: just look for closing tags and extract what's after
        # Qwen3 uses </think> or </think> as closing tag
        patterns = [
            r'</\\think>',  # Escaped backslash
            r'</think>',    # Regular closing tag
            r'</thinking>',
            r'</reasoning>',
            r'</think>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                # Keep only what's after the closing tag
                text = text[match.end():].strip()
                break
        
        # Remove any remaining tag content (in case tags are still present)
        text = re.sub(r'<[^>]*think[^>]*>.*?</[^>]*think[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]*reasoning[^>]*>.*?</[^>]*reasoning[^>]*>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Quick cleanup
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def _extract_after_reasoning(self, text: str) -> str:
        """
        Extract content that comes after reasoning blocks
        """
        import re
        # Find the last closing reasoning tag and return everything after it
        match = re.search(r'</(?:think|thinking|reasoning|redacted_reasoning)>', text, flags=re.IGNORECASE)
        if match:
            return text[match.end():]
        # If no closing tag, check if we have content that's clearly not reasoning
        # (starts with capital letter, not "Okay" or similar reasoning patterns)
        if re.match(r'^[A-Z][^<]*$', text.strip(), re.MULTILINE):
            return text
        return ""
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set a persistent system prompt
        
        Args:
            system_prompt: System prompt to use
        """
        # Remove existing system messages from history
        self.conversation_history = [
            msg for msg in self.conversation_history 
            if msg.get('role') != 'system'
        ]
        # Add system prompt as first message
        self.conversation_history.insert(0, {"role": "system", "content": system_prompt})
    
    def limit_history(self, max_messages: int = 20):
        """
        Limit conversation history to prevent it from growing too large
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        if len(self.conversation_history) > max_messages:
            # Keep system message if present, then keep last N messages
            system_msgs = [msg for msg in self.conversation_history if msg.get('role') == 'system']
            other_msgs = [msg for msg in self.conversation_history if msg.get('role') != 'system']
            self.conversation_history = system_msgs + other_msgs[-max_messages:]

