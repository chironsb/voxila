"""
Audio recording and playback module
"""
import pyaudio
import wave
import os
import threading
import sys
from typing import Optional


class AudioRecorder:
    """Handles audio recording from microphone"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        
    def record_to_file(self, file_path: str, duration: Optional[float] = None, stop_on_enter: bool = False) -> str:
        """
        Record audio from microphone to a file
        
        Args:
            file_path: Path to save the recorded audio
            duration: Maximum duration in seconds (None for manual stop)
            stop_on_enter: If True, stop recording when Enter is pressed
            
        Returns:
            Path to the recorded file
        """
        if stop_on_enter:
            print("Recording... (Press ENTER to stop)")
        elif duration:
            print(f"Recording... (will stop after {duration} seconds or press ENTER)")
        else:
            print("Recording... (Press ENTER to stop)")
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        stop_recording = threading.Event()
        
        def wait_for_enter():
            """Wait for Enter key press"""
            try:
                input()  # Wait for Enter
                stop_recording.set()
            except:
                pass
        
        # Start thread to wait for Enter if needed
        enter_thread = None
        if stop_on_enter or duration is None:
            enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
            enter_thread.start()
        
        try:
            if duration:
                # Record for specified duration or until Enter is pressed
                max_chunks = int(self.sample_rate / self.chunk_size * duration)
                for i in range(max_chunks):
                    if stop_recording.is_set():
                        break
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
            else:
                # Record until Enter is pressed
                while not stop_recording.is_set():
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            if stop_recording.is_set():
                print("\nRecording stopped.")
        
        # Save to file
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return file_path
    
    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()


class AudioPlayer:
    """Handles audio playback"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
    
    def play_file(self, file_path: str):
        """
        Play an audio file
        
        Args:
            file_path: Path to the audio file to play
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        wf = wave.open(file_path, 'rb')
        
        stream = self.audio.open(
            format=self.audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        
        stream.stop_stream()
        stream.close()
        wf.close()
    
    def cleanup(self):
        """Clean up audio resources"""
        self.audio.terminate()

