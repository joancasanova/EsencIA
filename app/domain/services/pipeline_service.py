# domain/services/pipeline_service.py

from typing import Any, Dict, List, Optional, Set, Tuple, Union, overload
import re
import logging
import itertools
import threading

from config.settings import MAX_PIPELINE_STEPS, MAX_VARIATIONS, MAX_CACHED_MODELS
from domain.model.entities.pipeline import PipelineStep, PipelineExecutionContext
from domain.model.entities.generation import GenerateTextRequest, GeneratedResult
from domain.model.entities.parsing import ParseRequest, ParseResult
from domain.model.entities.verification import VerificationMethod, VerificationSummary, VerifyRequest

from domain.services.parse_service import ParseService
from domain.services.verifier_service import VerifierService
from domain.services.generate_service import GenerateService

logger = logging.getLogger(__name__)

class PlaceholderDict(dict):
    """
    A custom dictionary that returns the placeholder itself if a key is not found.
    For example, if the key 'name' is not in the dictionary, it will return '{name}'.
    """
    def __missing__(self, key):
        return f"{{{key}}}"


class PipelineService:
    """
    Service for executing multi-step pipelines with generation, parsing, and verification.
    """

    def __init__(self, default_model_name: str):
        """
        Initializes the PipelineService with parsing, generation, and verification services.

        Args:
            default_model_name: The name of the language model to be used by default for all steps.
        """
        self.default_model_name = default_model_name

        self.parse_service = ParseService()

        # Cache for GenerateService instances with thread-safe access and LRU eviction
        self._generate_services_cache: Dict[str, GenerateService] = {}
        self._cache_order: List[str] = []  # LRU order: oldest first, newest last
        self._cache_lock = threading.Lock()

        # Initialize default service
        self._get_generate_service(default_model_name)

        # Current execution context (created fresh for each run_pipeline call)
        self._context: Optional[PipelineExecutionContext] = None
        self._context_lock = threading.Lock()

    @property
    def confirmed_references(self) -> List[Dict[str, str]]:
        """Returns confirmed references from the current execution context."""
        return list(self._context.confirmed_references) if self._context else []

    @property
    def to_verify_references(self) -> List[Dict[str, str]]:
        """Returns references needing verification from the current execution context."""
        return list(self._context.to_verify_references) if self._context else []

    @property
    def global_references(self) -> Dict[str, str]:
        """Returns global references from the current execution context."""
        return dict(self._context.global_references) if self._context else {}

    @global_references.setter
    def global_references(self, value: Dict[str, str]) -> None:
        """
        Sets global references for the current execution context.

        Thread-safe setter that ensures context exists before setting references.
        If no context exists, creates a new one. Uses immutable context pattern.
        """
        with self._context_lock:
            if self._context is None:
                self._context = PipelineExecutionContext(global_references=value)
            else:
                self._context = self._context.with_global_references(value)

    def run_pipeline(self, steps: List[PipelineStep]) -> None:
        """
        Executes the pipeline by processing each step sequentially.

        Thread-safe: each execution creates its own context and uses locks
        to prevent race conditions when accessing shared state.

        Args:
            steps: A list of PipelineStep objects defining the pipeline's steps.

        Raises:
            ValueError: If the number of steps exceeds MAX_PIPELINE_STEPS.
        """
        # Validate pipeline size
        if len(steps) > MAX_PIPELINE_STEPS:
            raise ValueError(
                f"Pipeline exceeds maximum steps limit ({MAX_PIPELINE_STEPS}). "
                f"Got {len(steps)} steps."
            )

        # Create fresh execution context for this run with thread-safe access
        # Preserve existing global_references if they were set before run_pipeline
        with self._context_lock:
            existing_global_refs = self._context.global_references if self._context else {}
            self._context = PipelineExecutionContext(global_references=existing_global_refs)

        step_number = 0  # Initialize before loop to ensure it's defined in except block
        try:
            for step_number, step in enumerate(steps):
                self._validate_step_references(step, step_number)
                step_result = self._execute_step(step, step_number)
                self._store_result(step_number, step.type, step_result)

        except Exception as e:
            logger.error(f"Pipeline execution failed at step {step_number}: {str(e)}")
            raise

    def get_results(self) -> List[Dict]:
        """
        Returns the accumulated results of all executed steps.
        """
        if self._context is None:
            return []
        serializable_results = []
        for result in self._context.results:
            if result is None:
                continue
            step_type, step_data = result
            step_data_dicts = [
                self._serialize_step_item(item, step_type)
                for item in step_data
            ]
            serializable_results.append({
                "step_type": step_type,
                "step_data": step_data_dicts
            })
        return serializable_results

    def _serialize_step_item(self, item: Any, step_type: str) -> Any:
        """
        Serializes a single step result item to dictionary format.

        Handles different types of step results with their specific serialization methods.

        Args:
            item: The step result item to serialize
            step_type: The type of step ('generate', 'parse', 'verify')

        Returns:
            Serialized representation of the item (typically a dict)
        """
        # Most items have a to_dict method
        if hasattr(item, 'to_dict'):
            return item.to_dict()

        # Special handling for verify steps
        if step_type == 'verify' and hasattr(item, 'verification_summary'):
            return item.verification_summary.to_dict()

        # Special handling for parse steps
        if step_type == 'parse' and hasattr(item, 'parse_result'):
            return item.parse_result.to_list_of_dicts()

        # Return as-is if no serialization method found
        return item
    
    def _validate_step_references(self, step: PipelineStep, step_number: int) -> None:
        """
        Validates that the reference_step_numbers for a step are valid (exist and are before current step).

        Args:
            step: The PipelineStep object being validated.
            step_number: The index of the current step.

        Raises:
            ValueError: If any reference is invalid.
        """
        if step.uses_reference:
            if not step.reference_step_numbers and not self._context.global_references:
                raise ValueError(f"Step {step_number} uses references but no references are provided.")
            # Safe iteration: reference_step_numbers may be None if only global_references are used
            for ref_index in step.reference_step_numbers or []:
                if not (0 <= ref_index < step_number):
                    raise ValueError(f"Step {step_number} has an invalid reference to step {ref_index}. Must be a previous step.")

    def _execute_step(self, step: PipelineStep, step_number: int) -> List[Any]:
        """
        Executes the logic for a single pipeline step.

        Handles reference and verification checks. Returns an empty list if validations fail.

        Args:
            step: The PipelineStep object defining the current step.
            step_number: The index of the current step.

        Returns:
            A list of results from the step, or an empty list if validations fail.
        """
        if step.uses_reference and not self._validate_references(step.reference_step_numbers, step_number):
            return []
        
        if step.type == "generate":
            return self._execute_generate(step, step_number)
        elif step.type == "parse":
            return self._execute_parse(step, step_number)
        elif step.type == "verify":
            return self._execute_verify(step, step_number)
        else:
            logger.warning(f"Unknown step type: {step.type}")
            return []

    def _store_result(self, step_number: int, step_type: str, step_result: List[Any]) -> None:
        """
        Stores the results of a step in the context's results list.

        Uses immutable context pattern - creates a new context with the updated result.

        Args:
            step_number: The index of the current step.
            step_type: The type of the step (e.g., 'generate', 'parse', 'verify').
            step_result: The list of results from the step.
        """
        self._context = self._context.with_result(
            step_number=step_number,
            step_type=step_type,
            step_result=tuple(step_result)
        )

    def _validate_references(self, reference_step_numbers: Optional[List[int]], current_step_number: int) -> bool:
        """
        Checks if the referenced steps have valid results and are before the current step.

        Args:
            reference_step_numbers: A list of indices of steps being referenced (may be None).
            current_step_number: The index of the current step.

        Returns:
            True if all references are valid or None, False otherwise.
        """
        for ref_index in reference_step_numbers or []:
            if not (0 <= ref_index < current_step_number and ref_index < len(self._context.results) and self._context.results[ref_index]):
                return False
        return True

    def _execute_generate(self, step: PipelineStep, step_number: int) -> List[GeneratedResult]:
        """
        Executes a 'generate' step, handling prompt variations based on references.

        Args:
            step: The PipelineStep object for the generate step.
            step_number: The index of the current step.

        Returns:
            A list of GeneratedResult objects.
        """
        request: GenerateTextRequest = step.parameters

        reference_data = self._get_reference_data(step.reference_step_numbers, step_number)

        prompt_variations = self._generate_variations(
            system_prompt_template=request.system_prompt,
            user_prompt_template=request.user_prompt,
            reference_data=reference_data,
            other_attributes=None
        )

        all_results = []

        # Determine which service to use
        model_name = step.llm_config if step.llm_config else self.default_model_name
        generate_service = self._get_generate_service(model_name)

        for system_prompt, user_prompt, reference_dict in prompt_variations:
            results = generate_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_sequences=request.num_sequences,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )

            for result in results:
                result.reference_data = reference_dict
            all_results.extend(results)

        return all_results

    def _execute_parse(self, step: PipelineStep, step_number: int) -> List[ParseResult]:
        """
        Executes a 'parse' step.

        Args:
            step: The PipelineStep object for the parse step.
            step_number: The index of the current step.

        Returns:
            A list containing the ParseResult.
        """
        request: ParseRequest = step.parameters
        
        if not step.uses_reference:
            parse_result = self.parse_service.parse_text(
                text=request.text,
                rules=request.rules
            )

            filtered_parse_result = self.parse_service.filter_entries(
                parse_result=parse_result,
                filter_type=request.output_filter,
                n=request.output_limit,
                rules=request.rules
            )

            return [filtered_parse_result]
    
        reference_data = self._get_reference_data(step.reference_step_numbers, step_number)
        
        parse_results = []
        for _, step_type, step_results in reference_data: 
            if step_type == "generate":
                generated_result: GeneratedResult
                for generated_result in step_results:
                    text = generated_result.content
                    parse_result = self.parse_service.parse_text(
                        text=text,
                        rules=request.rules
                    )

                    filtered_parse_result = self.parse_service.filter_entries(
                        parse_result=parse_result,
                        filter_type=request.output_filter,
                        n=request.output_limit,
                        rules=request.rules
                    )
                    parse_results.append(filtered_parse_result)

        return parse_results


    def _execute_verify(self, step: PipelineStep, step_number: int) -> List[VerificationSummary]:
        """
        Executes a 'verify' step, handling method variations based on references.

        Args:
            step: The PipelineStep object for the verify step.
            step_number: The index of the current step.

        Returns:
            A list of VerificationSummary objects.
        """
        request: VerifyRequest = step.parameters

        # Determine which service to use for verification
        model_name = step.llm_config if step.llm_config else self.default_model_name

        # Create a verifier service with the appropriate model
        verifier_service = VerifierService(generate_service=self._get_generate_service(model_name))

        if not step.uses_reference:
            verification_summary = verifier_service.verify(
                methods=request.methods,
                required_for_confirmed=request.required_for_confirmed,
                required_for_review=request.required_for_review
            )
            return [verification_summary]


        reference_data = self._get_reference_data(step.reference_step_numbers, step_number)

        verify_requests_variations = self._create_methods_variations(request, reference_data)

        all_results = []
        for verify_request, reference_dict in verify_requests_variations:
            result = verifier_service.verify(
                methods=verify_request.methods,
                required_for_confirmed=verify_request.required_for_confirmed,
                required_for_review=verify_request.required_for_review
            )

            result.reference_data = reference_dict
            all_results.append(result)

            # Capture references based on status using immutable context pattern
            if result.final_status == "confirmed":
                self._context = self._context.with_confirmed_reference(reference_dict)
            elif result.final_status == "review":
                self._context = self._context.with_to_verify_reference(reference_dict)

        return all_results
    
    def _create_methods_variations(
        self,
        original_request: VerifyRequest,
        reference_data: List[Tuple[int, str, List[Any]]]
    ) -> List[Tuple[VerifyRequest, Dict[str, str]]]:
        """
        Generates variations of VerifyRequests by combining variations of VerificationMethods.

        Args:
            original_request: The original VerifyRequest.
            reference_data: Data from referenced steps.

        Returns:
            A list of tuples, each containing a new VerifyRequest with method variations and its reference data.
        """
        new_requests = []

        # Generate variations for each method
        methods_variations = [
            self._generate_variations(
                method.system_prompt,
                method.user_prompt,
                reference_data,
                {
                    "mode": method.mode,
                    "name": method.name,
                    "num_sequences": method.num_sequences,
                    "valid_responses": method.valid_responses,
                    "required_matches": method.required_matches,
                    "max_tokens": method.max_tokens,
                    "temperature": method.temperature
                }
            )
            for method in original_request.methods
        ]

        # Combine method variations using itertools.product
        for combination in itertools.product(*methods_variations):
            # Safety limit to prevent exponential explosion of combinations
            if len(new_requests) >= MAX_VARIATIONS:
                raise RuntimeError(
                    f"Maximum number of method combinations ({MAX_VARIATIONS}) exceeded. "
                    f"Reduce reference data or limit methods per verification step."
                )

            new_methods = []
            combined_reference_dict = {}
            for method_variation, reference_dict in combination:
                new_methods.append(method_variation)
                combined_reference_dict.update(reference_dict)

            new_request = VerifyRequest(
                methods=new_methods,
                required_for_confirmed=original_request.required_for_confirmed,
                required_for_review=original_request.required_for_review
            )
            new_requests.append((new_request, combined_reference_dict))

        return new_requests

    @overload
    def _generate_variations(
        self,
        system_prompt_template: str,
        user_prompt_template: str,
        reference_data: List[Tuple[int, str, List[Any]]],
        other_attributes: None = None
    ) -> List[Tuple[str, str, Dict[str, str]]]: ...

    @overload
    def _generate_variations(
        self,
        system_prompt_template: str,
        user_prompt_template: str,
        reference_data: List[Tuple[int, str, List[Any]]],
        other_attributes: Dict[str, Any]
    ) -> List[Tuple[VerificationMethod, Dict[str, str]]]: ...

    def _generate_variations(
        self,
        system_prompt_template: str,
        user_prompt_template: str,
        reference_data: List[Tuple[int, str, List[Any]]],
        other_attributes: Optional[Dict[str, Any]] = None
    ) -> Union[List[Tuple[str, str, Dict[str, str]]], List[Tuple[VerificationMethod, Dict[str, str]]]]:
        """
        Generates variations of prompts or methods by filling placeholders with reference data.
        Refactored to be used for both prompt and method variations.

        Args:
            system_prompt_template: The system prompt template.
            user_prompt_template: The user prompt template.
            reference_data: Data from referenced steps.
            other_attributes: Other attributes to include in the generated object (for methods).

        Returns:
            A list of tuples, each containing a generated object (prompt or method) and its reference data.
        """
        variations = []

        def generate_combinations(
            index: int,
            system_prompt: str,
            user_prompt: str,
            current_reference_dict: Dict[str, str]
        ) -> None:
            """
            Recursive helper function to generate prompt/method combinations.
            """
            if not self._has_placeholders(system_prompt) and not self._has_placeholders(user_prompt):
                # Safety limit check before adding new variation
                if len(variations) >= MAX_VARIATIONS:
                    raise RuntimeError(
                        f"Maximum number of variations ({MAX_VARIATIONS}) exceeded. "
                        f"Reduce reference data or limit results per step."
                    )
                if other_attributes:
                    # Create a VerificationMethod object
                    variations.append(
                        (
                            VerificationMethod(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                **other_attributes
                            ),
                            current_reference_dict
                        )
                    )
                else:
                    # Create a prompt tuple
                    variations.append((system_prompt, user_prompt, current_reference_dict))
                return

            if index == len(reference_data):
                system_prompt, user_prompt, current_reference_dict = self._process_placeholders(system_prompt, user_prompt, self._context.global_references, current_reference_dict)
                # Safety limit check before adding new variation
                if len(variations) >= MAX_VARIATIONS:
                    raise RuntimeError(
                        f"Maximum number of variations ({MAX_VARIATIONS}) exceeded. "
                        f"Reduce reference data or limit results per step."
                    )
                if other_attributes:
                    variations.append(
                        (
                            VerificationMethod(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                **other_attributes
                            ),
                            current_reference_dict
                        )
                    )
                else:
                    variations.append((system_prompt, user_prompt, current_reference_dict))
                return

            ref_index, step_type, step_results = reference_data[index]
            if step_type == "generate":
                for generated_result in step_results:
                    content = f"output_{ref_index + 1}"
                    entry = {content: generated_result.content}
                    new_system_prompt, new_user_prompt, new_reference_dict = self._process_placeholders(system_prompt, user_prompt, entry, current_reference_dict.copy())
                    generate_combinations(index + 1, new_system_prompt, new_user_prompt, new_reference_dict)
            elif step_type == "parse":
                for parse_result in step_results:
                    for entry in parse_result.entries:
                        new_system_prompt, new_user_prompt, new_reference_dict = self._process_placeholders(system_prompt, user_prompt, entry, current_reference_dict.copy())
                        generate_combinations(index + 1, new_system_prompt, new_user_prompt, new_reference_dict)
            else:
                generate_combinations(index + 1, system_prompt, user_prompt, current_reference_dict)

        generate_combinations(
            index=0,
            system_prompt=system_prompt_template,
            user_prompt=user_prompt_template,
            current_reference_dict={}
        )
        return variations

    def _process_placeholders(
        self,
        system_prompt: str,
        user_prompt: str,
        references: Dict[str, str],
        reference_dict: Dict[str, str]
    ) -> Tuple[str, str, Dict[str, str]]:
        """
        Processes placeholders in prompts by replacing them with reference values.

        Args:
            system_prompt: The system prompt template with placeholders
            user_prompt: The user prompt template with placeholders
            references: Dictionary of reference key-value pairs to substitute
            reference_dict: Dictionary to track which references were actually used

        Returns:
            Tuple of (new_system_prompt, new_user_prompt, updated_reference_dict)
        """
        new_system_prompt = system_prompt
        new_user_prompt = user_prompt

        for reference_key, reference_value in references.items():
            new_system_prompt, replaced_flag_sys = self._replace_placeholders(new_system_prompt, {reference_key: reference_value})
            new_user_prompt, replaced_flag_usr = self._replace_placeholders(new_user_prompt, {reference_key: reference_value})

            if replaced_flag_sys or replaced_flag_usr:
                reference_dict[reference_key] = reference_value

        return new_system_prompt, new_user_prompt, reference_dict

    def _has_placeholders(self, text: str) -> bool:
        """
        Checks if a text contains any placeholders.

        Args:
            text: The text to check.

        Returns:
            True if the text contains at least one placeholder, False otherwise.
        """
        return bool(re.search(r"{[^{}]+}", text))

    def _get_placeholders(self, text: str) -> Set[str]:
        """
        Extracts all placeholder names from a text.

        Args:
            text: The text to extract placeholders from.

        Returns:
            A set of placeholder names found (without braces).
        """
        return set(re.findall(r"{([^{}]+)}", text))

    def _replace_placeholders(self, text: str, placeholders: Dict[str, str]) -> Tuple[str, bool]:
        """
        Replaces placeholders in a text given a dictionary of placeholders.

        If a placeholder is not found in the dictionary, the text keeps the placeholder form {placeholder}.

        Args:
            text: The text in which to search and replace placeholders.
            placeholders: A key-value dictionary of placeholders and their replacements.

        Returns:
            A tuple: (replaced_text, was_modified).
        """
        placeholder_dict = PlaceholderDict(placeholders)
        modified_text = text.format_map(placeholder_dict)
        was_replaced = modified_text != text
        return modified_text, was_replaced

    def _get_reference_data(
        self,
        reference_step_numbers: Optional[List[int]],
        current_step_number: int
    ) -> List[Tuple[int, str, List[Any]]]:
        """
        Retrieves the information from the referenced steps.

        Args:
            reference_step_numbers: A list of indices of steps being referenced (may be None).
            current_step_number: The index of the current step.

        Returns:
            A list of tuples (step_type, results) for each valid reference.
            Returns an empty list if reference_step_numbers is None or any reference is invalid.
        """
        reference_data = []
        for ref_index in reference_step_numbers or []:
            if 0 <= ref_index < current_step_number and ref_index < len(self._context.results) and self._context.results[ref_index]:
                step_type, results = self._context.results[ref_index]
                reference_data.append((ref_index, step_type, results))
            else:
                logger.warning(f"Reference {ref_index} not found or invalid for step {current_step_number}. Returning empty result.")
                return []
        return reference_data

    def _get_generate_service(self, model_name: str) -> GenerateService:
        """
        Retrieves or creates a GenerateService for the specified model.

        Implements LRU (Least Recently Used) cache eviction to prevent CUDA OOM
        when using multiple models. When cache exceeds MAX_CACHED_MODELS, the
        least recently used model is evicted and its GPU memory is freed.

        Thread-safe access to the shared cache of GenerateService instances.
        """
        with self._cache_lock:
            if model_name in self._generate_services_cache:
                # Move to end of LRU order (most recently used)
                if model_name in self._cache_order:
                    self._cache_order.remove(model_name)
                self._cache_order.append(model_name)
                return self._generate_services_cache[model_name]

            # Evict oldest model if cache is full
            while len(self._generate_services_cache) >= MAX_CACHED_MODELS:
                self._evict_oldest_model()

            # Create and cache new service
            logger.info(f"Loading model '{model_name}' into cache")
            self._generate_services_cache[model_name] = GenerateService(model_name)
            self._cache_order.append(model_name)
            return self._generate_services_cache[model_name]

    def _evict_oldest_model(self) -> None:
        """
        Evicts the least recently used model from cache and frees GPU memory.

        Must be called with _cache_lock held.
        """
        if not self._cache_order:
            return

        oldest_model = self._cache_order.pop(0)
        if oldest_model in self._generate_services_cache:
            logger.info(f"Evicting model '{oldest_model}' from cache to free memory")
            service = self._generate_services_cache.pop(oldest_model)
            self._free_model_memory(service)

    def _free_model_memory(self, service: GenerateService) -> None:
        """
        Frees GPU/CPU memory used by a GenerateService.

        Args:
            service: The GenerateService to free
        """
        try:
            # Delete model and tokenizer references
            if hasattr(service, 'model'):
                del service.model
            if hasattr(service, 'tokenizer'):
                del service.tokenizer

            # Force garbage collection and CUDA cache cleanup
            import gc
            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared after model eviction")
        except Exception as e:
            logger.warning(f"Error freeing model memory: {e}")

    def clear_model_cache(self) -> None:
        """
        Clears all cached models and frees GPU memory.

        Use this method when you need to free all GPU memory, for example
        before loading a very large model or when switching tasks.
        """
        with self._cache_lock:
            logger.info(f"Clearing model cache ({len(self._generate_services_cache)} models)")
            for model_name in list(self._generate_services_cache.keys()):
                service = self._generate_services_cache.pop(model_name)
                self._free_model_memory(service)
            self._cache_order.clear()
            logger.info("Model cache cleared")