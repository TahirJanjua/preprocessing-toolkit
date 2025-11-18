import librosa
import numpy as np
from typing import Tuple

class AudioPreprocessor:
    """
    Applies preprocessing techniques to audio signals.
    Supports time stretching, pitch shifting, and cascaded approaches.
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def load_audio(self, filepath: str, duration: float = 2.0) -> np.ndarray:
        """Load and normalize audio to fixed duration."""
        try:
            y, sr = librosa.load(filepath, sr=self.sr, duration=duration)
            # Pad or trim to exact duration
            target_length = int(self.sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            return y
        except Exception as e:
            raise ValueError(f"Error loading audio: {e}")
    
    def time_stretch(self, y: np.ndarray, rate: float) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        rate > 1: speed up, rate < 1: slow down
        """
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Pitch shift audio without changing speed.
        n_steps: semitones (-12 to +12 typical)
        """
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def apply_preprocessing(self, y: np.ndarray, technique: str, 
                           time_stretch_rate: float = 1.0,
                           pitch_shift_steps: int = 0) -> np.ndarray:
        """
        Apply preprocessing technique to audio.
        
        Args:
            y: Audio signal
            technique: 'none', 'time_stretch', 'pitch_shift', 'pitch_then_time', 'time_then_pitch'
            time_stretch_rate: Stretching factor
            pitch_shift_steps: Semitone shift
        """
        if technique == 'none':
            return y
        elif technique == 'time_stretch':
            return self.time_stretch(y, rate=time_stretch_rate)
        elif technique == 'pitch_shift':
            return self.pitch_shift(y, n_steps=pitch_shift_steps)
        elif technique == 'pitch_then_time':
            y = self.pitch_shift(y, n_steps=pitch_shift_steps)
            return self.time_stretch(y, rate=time_stretch_rate)
        elif technique == 'time_then_pitch':
            y = self.time_stretch(y, rate=time_stretch_rate)
            return self.pitch_shift(y, n_steps=pitch_shift_steps)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        return y / (np.max(np.abs(y)) + 1e-8)

