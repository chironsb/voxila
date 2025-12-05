"""
Automatic Speech Recognition (ASR) module using faster-whisper
"""
import os
from faster_whisper import WhisperModel
from typing import Optional


class ASR:
    """Speech-to-text using faster-whisper"""
    
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the ASR model
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu or cuda)
            compute_type: Compute type (int8, int8_float16, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        print(f"Loading Whisper model: {self.model_size} on {self.device}...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        print("Whisper model loaded successfully.")
    
    def transcribe(self, audio_file: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_file: Path to the audio file
            language: Language code (e.g., 'en', 'ro'). If None, auto-detect
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if self.model is None:
            self._load_model()
        
        print(f"Transcribing audio: {audio_file}")
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_file,
            language=language,
            beam_size=5
        )
        
        # Combine all segments
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        
        transcription = transcription.strip()
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        return transcription
    
    def transcribe_stream(self, audio_file: str, language: Optional[str] = None):
        """
        Transcribe audio file with streaming (yields segments as they're processed)
        
        Args:
            audio_file: Path to the audio file
            language: Language code (e.g., 'en', 'ro'). If None, auto-detect
            
        Yields:
            Text segments as they're transcribed
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if self.model is None:
            self._load_model()
        
        segments, info = self.model.transcribe(
            audio_file,
            language=language,
            beam_size=5
        )
        
        for segment in segments:
            yield segment.text

