"""
Base Speech-to-Text (STT) Interface

This module defines the abstract base class for Speech-to-Text implementations.
Students should implement the concrete STT class by inheriting from this base class.

Recommended implementation: Deepgram API (free tier available)
Alternative options: OpenAI Whisper, AssemblyAI, or any other STT service
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    
    This class defines the interface that all STT implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT service.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
                   Example: {"api_key": "your_api_key", "model": "nova-2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the STT service (setup API clients, load models, etc.).
        This method should be called before using the STT service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data as bytes
            **kwargs: Additional parameters specific to the STT implementation
                     (e.g., language, model, formatting options)
        
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the STT service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the STT service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class STTService(BaseSTT):
    """
    Generic STT implementation template.
    
    Students should complete this implementation using their chosen STT service or pretrained model.
    
    API-based options:
    - Deepgram API (free tier, high accuracy): pip install deepgram-sdk
    - AssemblyAI (API-based): pip install assemblyai
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Speech: pip install google-cloud-speech
    
    Pretrained model options (local inference):
    - OpenAI Whisper: pip install openai-whisper (various sizes: tiny, base, small, medium, large)
    - Wav2Vec2 models: pip install transformers torch (Facebook's pretrained models)
    - SpeechRecognition + offline engines: pip install SpeechRecognition pocketsphinx
    - Vosk models: pip install vosk (lightweight, supports many languages)
    - Coqui STT: pip install coqui-stt (open-source, pretrained models available)
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        # TODO: Initialize your chosen STT client/model
        # API-based examples:
        # - For Deepgram: from deepgram import DeepgramClient
        # - For AssemblyAI: import assemblyai
        # Pretrained model examples:
        # - For Whisper: import whisper
        # - For Transformers: from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        # - For Vosk: import vosk
    
    async def initialize(self) -> None:
        """
        TODO: Implement STT service initialization.
        
        Steps:
        1. Get API key from config (if using API service)
        2. Create client instance or load model
        3. Set is_initialized to True
        4. Optionally test the connection
        
        Example for API-based services:
        ```python
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("API key not provided")
        self.client = YourSTTClient(api_key)
        ```
        
        Example for pretrained models (local):
        ```python
        # Whisper
        model_name = self.config.get("model", "base")
        self.client = whisper.load_model(model_name)
        
        # Wav2Vec2 with Transformers
        model_name = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.client = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Vosk
        model_path = self.config.get("model_path", "path/to/vosk-model")
        self.client = vosk.Model(model_path)
        ```
        """
        try:
            import whisper
            import os
            try:
                import imageio_ffmpeg
                ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
                os.environ["PATH"] += os.pathsep + ffmpeg_path
            except ImportError:
                pass

            model_name = self.config.get("model", "base")
            print(f"Loading Whisper model: '{model_name}' (first run may download the model)...")
            self.client = whisper.load_model(model_name)
            self.is_initialized = True
            print(f"Whisper STT initialized successfully with model: {model_name}")
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )
        except Exception as e:
            self.is_initialized = False
            raise RuntimeError(f"STT initialization failed: {str(e)}")
    
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text using OpenAI Whisper.

        Input: audio_bytes (bytes) - Raw audio data in any supported format
        Output: str - Transcribed text

        Uses asyncio.to_thread to avoid blocking the event loop during
        CPU-intensive Whisper inference.
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized. Call initialize() first.")

        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")

        if len(audio_bytes) < 100:
            raise ValueError("Audio data is too short to contain speech")

        import asyncio

        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, **kwargs)

    def _transcribe_sync(self, audio_bytes: bytes, **kwargs) -> str:
        """Synchronous transcription helper — runs in a thread pool."""
        import io
        import os
        import tempfile
        import numpy as np

        # ------- Strategy 1: decode in-memory with soundfile -------
        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_data.size == 0 or sample_rate <= 0:
                raise ValueError("Invalid audio data")

            # Convert stereo to mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to 16 kHz (Whisper requirement)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                new_length = int(audio_data.shape[0] * target_sample_rate / sample_rate)
                if new_length <= 0:
                    raise ValueError("Invalid audio length after resampling")
                audio_data = np.interp(
                    np.linspace(0, audio_data.shape[0], num=new_length, endpoint=False),
                    np.arange(audio_data.shape[0]),
                    audio_data,
                ).astype("float32")

            result = self.client.transcribe(audio_data, fp16=False)
            transcription = result.get("text", "").strip()

            if transcription:
                print(f"Transcription: '{transcription}'")
            else:
                print("Whisper returned empty transcription (no speech detected)")

            return transcription

        except Exception as decode_error:
            _initial_decode_error = str(decode_error)
            print(f"In-memory decode failed ({_initial_decode_error}), falling back to temp file...")

        # ------- Strategy 2: write to temp file, let Whisper/ffmpeg handle it -------
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=os.getcwd()
            ) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            result = self.client.transcribe(tmp_path, fp16=False)
            transcription = result.get("text", "").strip()

            if transcription:
                print(f"Transcription: '{transcription}'")
            else:
                print("Whisper returned empty transcription (no speech detected)")

            return transcription

        except FileNotFoundError as e:
            raise RuntimeError(
                "Transcription failed: ffmpeg not found. "
                "Install ffmpeg (https://ffmpeg.org/download.html) and make sure it is on your PATH, "
                "or provide 16 kHz WAV audio."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Transcription failed: {str(e)} (initial decode error: {_initial_decode_error})"
            ) from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    
    async def cleanup(self) -> None:
        """
        TODO: Implement cleanup logic.
        
        Steps:
        1. Close any open connections
        2. Clear client instance
        3. Set is_initialized to False
        """
        try:
            self.client = None
            self.is_initialized = False
            print("STT cleanup completed")
        except Exception as e:
            print(f"STT cleanup error: {str(e)}")