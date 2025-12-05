"""
Text-to-Speech (TTS) module using Piper TTS
"""
import os
import subprocess
import tempfile
from typing import Optional


class TTS:
    """Text-to-speech using Piper TTS"""
    
    def __init__(self, piper_binary_path: str, model_path: str):
        """
        Initialize the TTS system
        
        Args:
            piper_binary_path: Path to the Piper binary
            model_path: Path to the Piper model (.onnx file)
        """
        self.piper_binary_path = piper_binary_path
        self.model_path = model_path
        
        if not os.path.exists(self.piper_binary_path):
            raise FileNotFoundError(
                f"Piper binary not found: {self.piper_binary_path}\n"
                "Please download Piper from: https://github.com/rhasspy/piper/releases"
            )
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper model not found: {self.model_path}\n"
                "Please download a model from: https://huggingface.co/rhasspy/piper-voices"
            )
    
    def synthesize(self, text: str, output_file: Optional[str] = None) -> str:
        """
        Synthesize speech from text
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file. If None, creates a temp file
            
        Returns:
            Path to the generated audio file
        """
        if output_file is None:
            # Create temporary file
            fd, output_file = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        
        try:
            # Run Piper TTS
            cmd = [
                self.piper_binary_path,
                '--model', self.model_path,
                '--output_file', output_file
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper TTS failed: {stderr}")
            
            if not os.path.exists(output_file):
                raise RuntimeError(f"Piper TTS did not create output file: {output_file}")
            
            return output_file
            
        except Exception as e:
            # Clean up temp file on error
            if output_file and os.path.exists(output_file) and output_file.startswith('/tmp'):
                try:
                    os.remove(output_file)
                except:
                    pass
            raise
    
    def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Synthesize speech from text and return as bytes
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (WAV format)
        """
        temp_file = self.synthesize(text)
        try:
            with open(temp_file, 'rb') as f:
                audio_data = f.read()
            return audio_data
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

