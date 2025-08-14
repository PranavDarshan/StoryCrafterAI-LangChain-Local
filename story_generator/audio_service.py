import whisper
import logging
import tempfile
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioService:
    def __init__(self):
        logger.info("Initializing Whisper for audio transcription")
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper model for CPU"""
        try:
            # Use the tiny model for fastest CPU processing
            model_size = "tiny"  # Options: tiny, base, small, medium, large
            
            logger.info(f"Loading Whisper model: {model_size}")
            
            # Load with CPU-specific settings
            self.model = whisper.load_model(
                model_size,
                device="cpu"
            )
            
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Whisper"""
        if not self.model:
            logger.error("Whisper model not available")
            return None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                # Write uploaded file to temporary file
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            logger.info("Starting audio transcription...")
            
            # Transcribe audio with CPU-optimized settings
            result = self.model.transcribe(
                tmp_file_path,
                fp16=False,  # Disable fp16 for CPU
                verbose=False
            )
            
            transcription = result["text"].strip()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            logger.info(f"Audio transcribed successfully: {transcription[:100]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Clean up temp file if it exists
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            return None
    
    def get_supported_formats(self):
        """Return list of supported audio formats"""
        return ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac']
    
    def is_audio_supported(self):
        """Check if audio processing is available"""
        return self.model is not None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model:
            return {
                'model_available': True,
                'model_type': 'whisper-tiny',
                'device': 'cpu',
                'supported_languages': 'multilingual'
            }
        return {
            'model_available': False,
            'error': 'Whisper model not loaded'
        }