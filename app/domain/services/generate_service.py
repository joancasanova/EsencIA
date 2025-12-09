# domain/services/generate_service.py
"""
Servicio de generacion de texto con LLMs.

Este servicio implementa la filosofia de EsencIA de ejecucion local:
- Valida recursos del sistema ANTES de cargar modelos
- Proporciona mensajes de error claros sobre limitaciones de hardware
- Sugiere alternativas cuando un modelo no puede ejecutarse
"""

import logging
import re
import socket
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from domain.model.entities.generation import GeneratedResult, GenerationMetadata
from config.settings import validate_temperature, validate_max_tokens
from infrastructure.system_resources import (
    ModelCompatibilityChecker,
    get_system_info,
    get_model_requirements
)

logger = logging.getLogger(__name__)

# Timeout for model downloads (seconds)
MODEL_DOWNLOAD_TIMEOUT = 300  # 5 minutes


class InsufficientResourcesError(Exception):
    """
    Error lanzado cuando no hay recursos suficientes para cargar un modelo.

    Esta excepcion es parte de la filosofia de EsencIA de informar proactivamente
    sobre limitaciones de hardware antes de intentar operaciones costosas.

    Attributes:
        model_name: Nombre del modelo que se intento cargar
        required_vram: VRAM estimada necesaria (GB)
        available_vram: VRAM disponible (GB)
        required_ram: RAM estimada necesaria (GB)
        available_ram: RAM disponible (GB)
        suggestions: Lista de modelos alternativos sugeridos
    """

    def __init__(
        self,
        message: str,
        model_name: str = "",
        required_vram: float = 0,
        available_vram: float = 0,
        required_ram: float = 0,
        available_ram: float = 0,
        suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.model_name = model_name
        self.required_vram = required_vram
        self.available_vram = available_vram
        self.required_ram = required_ram
        self.available_ram = available_ram
        self.suggestions = suggestions or []


class GenerateService:
    """
    Text generation service using Hugging Face Transformers models.

    Handles:
    - Pre-validation of system resources before loading models
    - Model and tokenizer initialization
    - Prompt formatting for different model types
    - Text generation with configurable parameters
    - Response post-processing and cleanup
    - Metadata collection and tracking

    Supports both base models and instruction-tuned models.

    Philosophy:
        EsencIA esta disenado para ejecutarse en hardware de consumo.
        Este servicio valida ANTES de cargar un modelo si los recursos
        disponibles son suficientes, proporcionando feedback claro
        y sugerencias de modelos alternativos si es necesario.
    """

    def __init__(self, model_name: str, skip_resource_check: bool = False):
        """
        Initialize text generation service.

        Args:
            model_name: Hugging Face model identifier or local path
            skip_resource_check: Skip pre-validation of resources (not recommended)

        Raises:
            ValueError: If model_name is invalid
            InsufficientResourcesError: If system resources are insufficient
            RuntimeError: If model loading fails
            ConnectionError: If network timeout occurs during download
            MemoryError: If CUDA out of memory during loading
        """
        # Validate model_name
        if not model_name or not model_name.strip():
            logger.error("Model name cannot be empty")
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name
        self.instruct_mode = "instruct" in model_name.lower()

        # Pre-validate resources before attempting to load model
        if not skip_resource_check:
            self._validate_resources(model_name)

        logger.info(f"Initializing generator with model '{model_name}'.")

        # Determine device based on compatibility check
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            # Set socket timeout for model downloads
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(MODEL_DOWNLOAD_TIMEOUT)

            try:
                # Initialize tokenizer and model
                # SECURITY: trust_remote_code=False prevents execution of arbitrary code from model repos
                logger.debug("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=False
                )

                logger.debug("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=False
                )

                # Move model to device with CUDA OOM handling
                logger.debug(f"Moving model to {self.device}...")
                self.model.to(self.device)

                # Set model to evaluation mode
                self.model.eval()

                logger.info(f"Successfully loaded model on {self.device.upper()}")
            finally:
                # Restore original timeout
                socket.setdefaulttimeout(original_timeout)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory while loading model: {e}")
            # Try to free CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Provide detailed error with suggestions
            requirements = get_model_requirements(model_name)
            resources = get_system_info()
            raise InsufficientResourcesError(
                f"GPU memory insuficiente para cargar '{model_name}'.\n"
                f"Requerido: ~{requirements.estimated_vram_gb}GB VRAM\n"
                f"Disponible: ~{resources.available_vram_gb:.1f}GB VRAM\n"
                f"Sugerencia: Use un modelo mas pequeno como 'Qwen/Qwen2.5-0.5B-Instruct'",
                model_name=model_name,
                required_vram=requirements.estimated_vram_gb,
                available_vram=resources.available_vram_gb,
                suggestions=["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
            ) from e
        except (socket.timeout, TimeoutError) as e:
            logger.error(f"Network timeout while downloading model '{model_name}': {e}")
            raise ConnectionError(
                f"Model download timed out after {MODEL_DOWNLOAD_TIMEOUT}s. "
                f"Check your internet connection or try again later."
            ) from e
        except OSError as e:
            # Covers file not found, permission errors, etc.
            logger.error(f"OS error loading model '{model_name}': {e}")
            raise RuntimeError(
                f"Could not load model '{model_name}'. "
                f"Verify the model name or path is correct."
            ) from e
        except InsufficientResourcesError:
            # Re-raise without wrapping
            raise
        except Exception as e:
            logger.exception(f"Unexpected error loading model '{model_name}'")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _validate_resources(self, model_name: str) -> None:
        """
        Valida que los recursos del sistema sean suficientes para el modelo.

        Args:
            model_name: Nombre del modelo a validar

        Raises:
            InsufficientResourcesError: Si los recursos son insuficientes
        """
        checker = ModelCompatibilityChecker()
        result = checker.check(model_name)

        if not result.is_compatible:
            requirements = get_model_requirements(model_name)
            resources = checker.get_system_resources()

            # Obtener modelos recomendados
            recommended = checker.get_recommended_models()
            suggestions = [m["model_name"] for m in recommended[:3]]

            raise InsufficientResourcesError(
                result.error_message or f"Recursos insuficientes para '{model_name}'",
                model_name=model_name,
                required_vram=requirements.estimated_vram_gb,
                available_vram=resources.available_vram_gb,
                required_ram=requirements.estimated_ram_gb,
                available_ram=resources.available_ram_gb,
                suggestions=suggestions
            )

        # Log warnings if any
        for warning in result.warnings:
            logger.warning(f"Advertencia de recursos: {warning}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        num_sequences: int = 1,
        max_tokens: int = 200,
        temperature: float = 1.0
    ) -> List[GeneratedResult]:
        """
        Generate text sequences from prompts.

        Args:
            system_prompt: Context/instructions for the model
            user_prompt: Specific input/question to generate response for
            num_sequences: Number of variations to generate (default: 1)
            max_tokens: Maximum length of generated text (default: 100)
            temperature: Sampling randomness (0.0=deterministic, 1.0=default)

        Returns:
            List[GeneratedResult]: Generated text sequences with metadata

        Raises:
            ValueError: If temperature or max_tokens are out of valid range
            Exception: If generation fails
        """
        # Validate parameters before generation
        validate_temperature(temperature)
        validate_max_tokens(max_tokens)

        logger.debug(f"Generating with system prompt: {system_prompt[:50]}...")
        start_time = datetime.now()
        
        try:
            # Format prompt based on model type
            if self.instruct_mode:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = f"{system_prompt}\n{user_prompt}"

            # Tokenize and generate
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

            # Generate with CUDA OOM protection
            # temperature=0 means greedy decoding (do_sample=False)
            # temperature>0 means sampling with that temperature
            use_sampling = temperature > 0

            with torch.no_grad():  # Disable gradient computation for inference
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "num_return_sequences": num_sequences,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }

                if use_sampling:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = temperature
                else:
                    generate_kwargs["do_sample"] = False

                outputs = self.model.generate(**generate_kwargs)

            # Process outputs
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return self._create_results(decoded_outputs, prompt, system_prompt, user_prompt, temperature, start_time)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during generation: {e}")
            # Try to free CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MemoryError(
                f"Not enough GPU memory for generation with {num_sequences} sequences. "
                f"Try reducing num_sequences or max_tokens, or use CPU mode."
            ) from e
        except ValueError as e:
            logger.error(f"Invalid generation parameters: {e}")
            raise ValueError(f"Text generation failed due to invalid parameters: {e}") from e
        except RuntimeError as e:
            # Catch model execution errors that aren't CUDA OOM
            logger.error(f"Model runtime error during generation: {e}")
            raise RuntimeError(f"Model execution failed: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during text generation: {type(e).__name__}")
            raise RuntimeError(f"Text generation failed: {e}") from e

    def get_token_count(self, text: str) -> int:
        """
        Calculate token count for a given text string.

        Args:
            text: Input text to analyze

        Returns:
            int: Number of tokens in the text

        Raises:
            ValueError: If text is invalid or cannot be tokenized
            RuntimeError: If tokenizer encounters an error
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")

        try:
            return len(self.tokenizer.encode(text))
        except ValueError as e:
            logger.error(f"Invalid text for tokenization: {e}")
            raise ValueError(f"Cannot tokenize text: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected tokenization error: {type(e).__name__}")
            raise RuntimeError(f"Token counting failed: {e}") from e

    def _create_results(
        self,
        outputs: List[str],
        prompt: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        start_time: datetime
    ) -> List[GeneratedResult]:
        """
        Package raw outputs into GeneratedResult objects with metadata.

        Args:
            outputs: List of raw generated text strings
            prompt: Formatted prompt used for generation
            system_prompt: Original system prompt
            user_prompt: Original user prompt
            temperature: Temperature used for generation
            start_time: Generation start time for metrics

        Returns:
            List[GeneratedResult]: Structured results with metadata
        """
        results: List[GeneratedResult] = []
        for output in outputs:
            content = self._process_output(output, prompt)
            metadata = GenerationMetadata(
                model_name=self.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                tokens_used=len(self.tokenizer.encode(content)),
                generation_time=(datetime.now() - start_time).total_seconds()
            )
            results.append(GeneratedResult(content=content.strip(), metadata=metadata))
        logger.debug(f"Generated {len(results)} sequences")
        return results

    def _extract_assistant_response(self, text: str) -> str:
        """
        Extract assistant's response from instruction-formatted text.

        Uses pattern matching to find content after 'assistant' marker.

        Args:
            text: Generated text possibly containing formatting markers

        Returns:
            str: Extracted assistant response or original text if no match

        Raises:
            RuntimeError: If regex processing fails
        """
        try:
            match = re.search(r"assistant\n(.*)", text, re.DOTALL)
            return match.group(1).strip() if match else text.strip()
        except re.error as e:
            logger.error(f"Regex error in response extraction: {e}")
            raise RuntimeError(f"Failed to extract assistant response: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error extracting response: {type(e).__name__}")
            return text.strip()  # Fallback to original text

    def _trim_response(self, prompt: str, output: str) -> str:
        """
        Remove prompt text from generated output.

        Handles cases where model doesn't exactly reproduce prompt.

        Args:
            prompt: Original prompt text
            output: Generated output text

        Returns:
            str: Output with prompt removed
        """
        return output[len(prompt):].strip() if output.startswith(prompt) else output.strip()

    def _process_output(self, output: str, prompt: str) -> str:
        """
        Select appropriate processing method based on model type.

        Args:
            output: Raw generated output text
            prompt: Original prompt used for generation

        Returns:
            str: Processed output text
        """
        return self._extract_assistant_response(output) if self.instruct_mode else self._trim_response(prompt, output)