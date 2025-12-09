# application/use_cases/parse_use_case.py

import logging
from domain.model.entities.parsing import ParseRequest, ParseResponse
from domain.services.parse_service import ParseService

logger = logging.getLogger(__name__)

class ParseRequestValidationError(Exception):
    """
    Specialized exception for invalid parsing requests.
    
    Raised when request parameters fail validation checks, containing
    specific details about the validation failure.
    """
    pass

class ParseUseCase:
    """
    Orchestrates text parsing operations using configured parsing rules.
    
    Responsibilities:
    1. Validate input parameters for parsing operations
    2. Coordinate parsing workflow through ParseService
    3. Apply output filtering based on request parameters
    4. Handle error conditions and error reporting
    """

    def __init__(self):
        """
        Initializes parser components with dependency injection.
        
        Creates a ParseService instance to handle core parsing logic,
        allowing for different parser implementations while maintaining
        consistent interface.
        """
        self.parse_service = ParseService()

    def execute(self, request: ParseRequest) -> ParseResponse:
        """
        Executes complete parsing workflow with validation and filtering.
        
        Process Flow:
        1. Validate request parameters
        2. Execute text parsing using configured rules
        3. Apply requested output filtering
        4. Package results with success metrics
        
        Args:
            request: Contains parsing parameters including:
                    - text: Input text to parse
                    - rules: List of ParseRule configurations
                    - output_filter: Result filtering strategy
                    - output_limit: Maximum results to return
        
        Returns:
            ParseResponse: Contains filtered parsing results
            
        Raises:
            ParseRequestValidationError: For invalid request parameters
            Exception: Propagates any errors from parsing implementation
        """
        logger.info("Starting parsing workflow execution")
        try:
            # Primary validation of request parameters
            self._validate_request(request)
        except ValueError as e:
            # Convert generic ValueError to domain-specific exception
            logger.error(f"Request validation failed: {str(e)}")
            raise ParseRequestValidationError(f"Invalid parse request: {e}") from e

        try:
            # Core parsing operation
            logger.debug("Initiating text parsing with %d rules", len(request.rules))
            parse_result = self.parse_service.parse_text(
                text=request.text,
                rules=request.rules
            )
            
            # Result post-processing
            logger.debug("Applying %s filter to results", request.output_filter)
            filtered_result = self.parse_service.filter_entries(
                parse_result,
                request.output_filter,
                request.output_limit,
                request.rules
            )

            logger.info("Parsing completed with %d valid entries", len(filtered_result.entries))
            return ParseResponse(parse_result=filtered_result)

        except ValueError as e:
            logger.error(f"Invalid parsing configuration: {e}")
            raise ParseRequestValidationError(f"Parsing failed due to invalid configuration: {e}") from e
        except Exception as e:
            logger.exception(f"Critical error during parsing operations: {type(e).__name__}")
            raise RuntimeError(f"Parsing operation failed: {e}") from e

    def _validate_request(self, request: ParseRequest) -> None:
        """
        Ensures request meets minimum validity requirements.

        Validation Criteria:
        1. Input text must contain non-whitespace content (if provided)
        2. At least one parsing rule must be provided
        3. All rules must have non-empty pattern definitions

        Args:
            request: Parse request to validate

        Raises:
            ValueError: With detailed message about validation failure
        """
        # Validate input text content (text can be None when used in pipeline with references)
        if request.text is not None and not request.text.strip():
            raise ValueError("Text input cannot be empty or whitespace")

        # Validate rules existence
        if not request.rules:
            raise ValueError("At least one parsing rule must be provided")

        # Validate individual rule completeness
        for rule in request.rules:
            if not rule.pattern:
                raise ValueError(f"Rule '{rule.name}' missing required pattern")