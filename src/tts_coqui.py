"""
Coqui TTS module (simpler alternative to OpenVoice)
Uses the TTS library which is already in requirements
"""
import os
from TTS.api import TTS as CoquiTTS
from audio import AudioPlayer


class CoquiTTSWrapper:
    """Coqui TTS wrapper for simple text-to-speech"""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", device: str = None):
        """
        Initialize Coqui TTS
        
        Args:
            model_name: TTS model name (see https://github.com/coqui-ai/TTS)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading Coqui TTS model: {model_name} on {device}...")
        try:
            self.tts = CoquiTTS(model_name=model_name, progress_bar=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load Coqui TTS model: {e}")
        self.audio_player = AudioPlayer()
        self.output_dir = './outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        print("Coqui TTS loaded successfully.")
    
    def synthesize(self, text: str, output_file: str = None):
        """
        Synthesize speech from text
        
        Args:
            text: Text to convert to speech
            output_file: Output file path (optional)
            
        Returns:
            Path to generated audio file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'tts_output.wav')
        
        try:
            self.tts.tts_to_file(text=text, file_path=output_file)
            return output_file
        except Exception as e:
            raise RuntimeError(f"Error during TTS generation: {e}")
    
    def synthesize_and_play(self, text: str):
        """
        Synthesize speech and play it
        
        Args:
            text: Text to synthesize
        """
        output_file = self.synthesize(text)
        self.audio_player.play_file(output_file)

