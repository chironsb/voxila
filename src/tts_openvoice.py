"""
OpenVoice TTS module (like the original project)
"""
import os
import torch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import BaseSpeakerTTS, ToneColorConverter
import se_extractor
from audio import AudioPlayer


class OpenVoiceTTS:
    """OpenVoice TTS with voice cloning"""
    
    def __init__(self, 
                 checkpoints_dir: str = "./checkpoints",
                 reference_voice: str = "./voxila.mp3",
                 device: str = None):
        """
        Initialize OpenVoice TTS
        
        Args:
            checkpoints_dir: Directory containing checkpoints
            reference_voice: Path to reference voice audio file
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None
        """
        if device is None:
            # Default to CPU to avoid GPU memory issues when Whisper is also using GPU
            device = 'cpu'
            # Uncomment to use GPU if you have enough VRAM:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        # Make paths absolute relative to project root
        # Get project root (parent of src directory)
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(src_dir)
        
        # Resolve checkpoints and reference voice paths
        if not os.path.isabs(checkpoints_dir):
            checkpoints_dir = os.path.join(project_root, checkpoints_dir.lstrip('./'))
        if not os.path.isabs(reference_voice):
            reference_voice = os.path.join(project_root, reference_voice.lstrip('./'))
        
        self.checkpoints_dir = checkpoints_dir
        self.reference_voice = reference_voice
        # Output directory relative to project root
        self.output_dir = os.path.join(project_root, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paths
        en_ckpt_base = os.path.join(checkpoints_dir, 'base_speakers', 'EN')
        ckpt_converter = os.path.join(checkpoints_dir, 'converter')
        
        if not os.path.exists(en_ckpt_base):
            raise FileNotFoundError(
                f"OpenVoice checkpoints not found: {en_ckpt_base}\n"
                "Please download checkpoints from: https://nordnet.blob.core.windows.net/bilde/checkpoints.zip\n"
                "Or set OPENVOICE_CHECKPOINTS in .env to point to your checkpoints directory."
            )
        
        if not os.path.exists(reference_voice):
            raise FileNotFoundError(
                f"Reference voice not found: {reference_voice}\n"
                "Please provide a reference audio file for voice cloning.\n"
                "Set OPENVOICE_REFERENCE in .env to point to your reference audio file."
            )
        
        print(f"Loading OpenVoice models on {device}...")
        
        # Load models
        self.en_base_speaker_tts = BaseSpeakerTTS(
            f'{en_ckpt_base}/config.json', 
            device=device
        )
        self.en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
        
        self.tone_color_converter = ToneColorConverter(
            f'{ckpt_converter}/config.json', 
            device=device
        )
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        # Load speaker embeddings
        self.en_source_default_se = torch.load(
            f'{en_ckpt_base}/en_default_se.pth'
        ).to(device)
        self.en_source_style_se = torch.load(
            f'{en_ckpt_base}/en_style_se.pth'
        ).to(device)
        
        self.audio_player = AudioPlayer()
        print("OpenVoice TTS loaded successfully.")
    
    def synthesize(self, text: str, output_file: str = None, style: str = "default"):
        """
        Synthesize speech with voice cloning
        
        Args:
            text: Text to synthesize
            output_file: Output file path (optional)
            style: Voice style ('default' or 'style')
            
        Returns:
            Path to generated audio file
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'output.wav')
        
        tts_model = self.en_base_speaker_tts
        source_se = self.en_source_default_se if style == 'default' else self.en_source_style_se
        speaker_wav = self.reference_voice
        
        try:
            # Extract speaker embedding from reference voice
            # Get project root for processed directory
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(src_dir)
            processed_dir = os.path.join(project_root, 'processed')
            
            target_se, audio_name = se_extractor.get_se(
                speaker_wav, 
                self.tone_color_converter, 
                target_dir=processed_dir, 
                vad=True
            )
            
            # Generate base TTS
            src_path = os.path.join(self.output_dir, 'tmp.wav')
            tts_model.tts(text, src_path, speaker=style, language='English')
            
            # Convert tone color
            encode_message = "@MyShell"
            self.tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_file,
                message=encode_message
            )
            
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Error during audio generation: {e}")
    
    def synthesize_and_play(self, text: str, style: str = "default"):
        """
        Synthesize speech and play it
        
        Args:
            text: Text to synthesize
            style: Voice style ('default' or 'style')
        """
        output_file = self.synthesize(text, style=style)
        self.audio_player.play_file(output_file)

