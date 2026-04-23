"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
Students should complete the implementation to connect all components.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


@dataclass
class TranscriptData:
    """Data structure for transcript information."""
    user_input: str
    agent_response: str


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    Students should complete the implementation to make it fully functional.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the audio support pipeline.
        
        Args:
            config: Pipeline configuration containing settings for all components
        """
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        """
        TODO: Initialize all pipeline components.
        
        Steps:
        1. Initialize STT service
        2. Initialize LLM agent
        3. Initialize TTS service
        4. Verify all components are ready
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            self.logger.info("Initializing Audio Support Pipeline...")

            self.logger.info("Initializing STT service (Whisper)...")
            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()
            self.logger.info("STT ready")

            self.logger.info("Initializing LLM Agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()
            self.logger.info("LLM Agent ready")

            self.logger.info("Initializing TTS service (Edge TTS)...")
            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()
            self.logger.info("TTS ready")

            if not self.stt.is_ready():
                raise RuntimeError("STT component failed to initialize")
            if not self.llm_agent.is_initialized:
                raise RuntimeError("LLM Agent failed to initialize")
            if not self.tts.is_ready():
                raise RuntimeError("TTS component failed to initialize")

            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully — all 3 components ready!")

        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        TODO: Process audio input through the complete pipeline.
        
        This is the main method that handles the STT -> LLM -> TTS flow.
        
        Args:
            audio_bytes: Input audio data
            **kwargs: Additional parameters for processing
            
        Returns:
            bytes: Response audio data
            
        Raises:
            RuntimeError: If pipeline is not initialized
            Exception: If processing fails at any stage
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            response_audio, _, _ = await self.process_audio_with_transcript(audio_bytes, **kwargs)
            return response_audio

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise

    async def process_audio_with_transcript(
        self,
        audio_bytes: bytes,
        **kwargs
    ) -> Tuple[bytes, TranscriptData, int]:
        """
        TODO: Implement enhanced audio processing that returns audio, transcript, and timing.

        Returns:
            Tuple[bytes, TranscriptData, int]: (response_audio, transcript_data, processing_time_ms)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        try:
            self.logger.info("Step 1/3: STT — converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            self.logger.info(f"STT result: '{text_input}'")

            llm_input = text_input
            if not text_input or not text_input.strip():
                llm_input = "I couldn't hear you clearly. Could you please repeat your question?"

            self.logger.info("Step 2/3: LLM — generating agent response...")
            agent_response = await self.llm_agent.process_query(llm_input, **kwargs)
            self.logger.info(
                f"LLM response: '{agent_response[:80]}...' "
                if len(agent_response) > 80
                else f"LLM response: '{agent_response}'"
            )

            self.logger.info("Step 3/3: TTS — synthesizing audio response...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info(f"Pipeline complete — {len(response_audio)} bytes of audio generated")

            transcript_data = self._create_transcript_data(text_input or "", agent_response)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            return response_audio, transcript_data, processing_time_ms

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        TODO: Process text input (useful for testing without STT).
        
        Args:
            text_input: Text query from user
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, bytes]: (agent_response_text, response_audio)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        try:
            self.logger.info(f"Processing text: '{text_input}'")

            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info(f"Agent response ready: {len(agent_response)} chars")

            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info(f"Audio generated: {len(response_audio)} bytes")

            return agent_response, response_audio

        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise

    async def process_text_with_timing(self, text_input: str, **kwargs) -> Tuple[str, int]:
        """
        TODO: Process text and capture processing time.

        Returns:
            Tuple[str, int]: (agent_response_text, processing_time_ms)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.perf_counter()
        try:
            agent_response, _ = await self.process_text(text_input, **kwargs)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return agent_response, processing_time_ms
        except Exception as e:
            self.logger.error(f"Timed text processing failed: {str(e)}")
            raise

    def _create_transcript_data(self, user_input: str, agent_response: str) -> TranscriptData:
        """Create structured transcript data."""
        return TranscriptData(user_input=user_input, agent_response=agent_response)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        TODO: Check the health status of all pipeline components.
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """
        TODO: Cleanup all pipeline resources.
        
        This method should be called when the pipeline is no longer needed.
        """
        self.logger.info("Cleaning up pipeline resources...")
        try:
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()

            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            self.logger.info("Pipeline cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """
    TODO: Factory function to create and initialize a pipeline.
    
    Args:
        stt_config: STT configuration
        llm_config: LLM configuration  
        tts_config: TTS configuration
        enable_logging: Whether to enable logging
        
    Returns:
        AudioSupportPipeline: Initialized pipeline instance
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage of the pipeline.
    Students can use this for testing their implementation.
    """
    async def main():
        # TODO: Example configuration - replace with your chosen services
        stt_config = {
            # Configure your chosen STT service
            "api_key": "your_stt_api_key",
            "model": "your_chosen_model"
        }
        
        llm_config = {
            # Configure your chosen LLM service
            "api_key": "your_llm_api_key",
            "model": "your_chosen_model",
            "temperature": 0.7
        }
        
        tts_config = {
            # Configure your chosen TTS service
            "api_key": "your_tts_api_key",
            "voice_id": "your_chosen_voice"
        }
        
        # TODO: Create and test pipeline
        # pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        
        # TODO: Test with text input
        # response_text, response_audio = await pipeline.process_text("Hello, I need help with my order")
        # print(f"Response: {response_text}")
        
        # TODO: Cleanup
        # await pipeline.cleanup()
        
        print("Pipeline example completed. Implement the TODOs to make it functional!")
    
    asyncio.run(main())