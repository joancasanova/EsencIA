# domain/services/parse_service.py

import logging
import re
from typing import List, Dict, Optional, Literal
from domain.model.entities.parsing import ParseResult, ParseRule, ParseMode
from config.settings import MAX_PARSE_ITERATIONS, MAX_MATCHES_PER_RULE, MAX_PARSE_ENTRIES

logger = logging.getLogger(__name__)

class ParseService:
    """
    Text parsing service implementing rule-based extraction.
    Handles both regex patterns and keyword-based extraction with boundary detection.
    """

    def parse_text(self, text: str, rules: List[ParseRule]) -> ParseResult:
        """
        Main parsing method that processes text through multiple rule-based stages.

        Args:
            text: Input text to parse
            rules: Ordered list of parsing rules to apply

        Returns:
            ParseResult: Structured parsing results with extracted values
        """
        logger.debug("Starting text parsing process")

        # Stage 1: Find all rule matches in text
        all_matches = self._find_all_matches(text, rules)

        # Stage 2: Sort matches by position and rule order
        all_matches.sort(key=lambda x: (x["start"], x["rule_idx"]))

        # Stage 3: Build structured entries from matches
        entries = self._build_entries_from_matches(all_matches, rules)

        logger.debug(f"Parsing completed with {len(entries)} entries")
        return ParseResult(entries=entries)

    def _find_all_matches(self, text: str, rules: List[ParseRule]) -> List[Dict]:
        """
        Find all matches for all rules in the text.

        Args:
            text: Input text to search
            rules: List of parsing rules to apply

        Returns:
            List of match dictionaries with start, end, rule info, and value
        """
        all_matches = []
        for rule_idx, rule in enumerate(rules):
            occurrences = self._find_all_occurrences(text, rule)
            for (start, end, matched_str) in occurrences:
                all_matches.append({
                    "start": start,
                    "end": end,
                    "rule_idx": rule_idx,
                    "rule_name": rule.name,
                    "value": matched_str
                })
        return all_matches

    def _build_entries_from_matches(
        self,
        all_matches: List[Dict],
        rules: List[ParseRule]
    ) -> List[Dict[str, str]]:
        """
        Build structured entries from sorted matches.

        Handles rule sequencing, duplicates, and missing rules with fallbacks.

        Args:
            all_matches: Sorted list of match dictionaries
            rules: Original parsing rules for fallback values

        Returns:
            List of complete entry dictionaries

        Raises:
            RuntimeError: If maximum entry limit is exceeded
        """
        entries: List[Dict[str, str]] = []
        current_entry: Dict[str, str] = {}
        rules_matched_in_entry = set()
        expected_rule_index = 0

        for match_info in all_matches:
            # Safety check: limit number of entries
            if len(entries) >= MAX_PARSE_ENTRIES:
                logger.warning(
                    f"Maximum entry limit ({MAX_PARSE_ENTRIES}) reached. "
                    f"Stopping entry construction."
                )
                break

            rule_idx = match_info["rule_idx"]
            rule_name = match_info["rule_name"]
            matched_str = match_info["value"]

            # Handle duplicate rule or out-of-order rule
            if self._should_finalize_entry(
                rule_name, rule_idx, rules_matched_in_entry, expected_rule_index
            ):
                self._finalize_entry(
                    current_entry, rules_matched_in_entry, rules, entries
                )
                expected_rule_index = 0

            # Add current match to entry
            expected_rule_index = self._add_match_to_entry(
                current_entry,
                rules_matched_in_entry,
                rule_idx,
                rule_name,
                matched_str,
                expected_rule_index,
                rules,
                entries
            )

        # Finalize any remaining partial entry (if under limit)
        if rules_matched_in_entry and len(entries) < MAX_PARSE_ENTRIES:
            self._finalize_entry(
                current_entry, rules_matched_in_entry, rules, entries
            )

        return entries

    def _should_finalize_entry(
        self,
        rule_name: str,
        rule_idx: int,
        rules_matched: set,
        expected_idx: int
    ) -> bool:
        """Check if current entry should be finalized before processing match."""
        # Duplicate rule in current entry
        if rule_name in rules_matched:
            return True
        # Rule appears before expected order
        if rule_idx < expected_idx:
            return True
        return False

    def _add_match_to_entry(
        self,
        current_entry: Dict[str, str],
        rules_matched: set,
        rule_idx: int,
        rule_name: str,
        matched_str: str,
        expected_idx: int,
        rules: List[ParseRule],
        entries: List[Dict[str, str]]
    ) -> int:
        """
        Add match to current entry, filling missing rules if needed.

        Returns:
            Updated expected rule index
        """
        # Expected rule in sequence
        if rule_idx == expected_idx:
            current_entry[rule_name] = matched_str
            rules_matched.add(rule_name)
            return expected_idx + 1

        # Rule appears after expected position - fill gaps
        elif rule_idx > expected_idx:
            self._fill_missing_rules(
                current_entry, rules_matched, expected_idx, rule_idx, rules
            )
            current_entry[rule_name] = matched_str
            rules_matched.add(rule_name)
            self._finalize_entry(current_entry, rules_matched, rules, entries)
            return 0

        return expected_idx

    def _fill_missing_rules(
        self,
        current_entry: Dict[str, str],
        rules_matched: set,
        start_idx: int,
        end_idx: int,
        rules: List[ParseRule]
    ) -> None:
        """Fill missing rules with fallback values."""
        for missing_idx in range(start_idx, end_idx):
            missing_rule = rules[missing_idx]
            fallback = missing_rule.fallback_value or f"missing_{missing_rule.name}"
            current_entry[missing_rule.name] = fallback
            rules_matched.add(missing_rule.name)

    def _finalize_entry(
        self,
        current_entry: Dict[str, str],
        rules_matched: set,
        rules: List[ParseRule],
        entries: List[Dict[str, str]]
    ) -> None:
        """Complete current entry by adding fallback values for missing rules."""
        for rule in rules:
            if rule.name not in rules_matched:
                fallback = rule.fallback_value or f"missing_{rule.name}"
                current_entry[rule.name] = fallback
        entries.append(dict(current_entry))
        current_entry.clear()
        rules_matched.clear()

    def _find_all_occurrences(self, text: str, rule: ParseRule) -> List[tuple]:
        """
        Find all occurrences of a rule's pattern in the text.

        Returns list of tuples containing:
        (start_index, end_index, matched_string)

        Raises:
            re.error: If regex pattern is invalid (logged and returns empty list)
            RuntimeError: If iteration or match limits are exceeded
        """
        logger.debug(f"Processing rule: {rule.name} ({rule.mode})")
        results = []

        if rule.mode == ParseMode.REGEX:
            # Regex pattern matching with full capture
            try:
                for match in re.finditer(rule.pattern, text):
                    # Check match limit
                    if len(results) >= MAX_MATCHES_PER_RULE:
                        logger.warning(
                            f"Rule '{rule.name}' exceeded maximum matches limit "
                            f"({MAX_MATCHES_PER_RULE}). Stopping extraction."
                        )
                        break

                    results.append((
                        match.start(),
                        match.end(),
                        match.group().strip()
                    ))
            except re.error as e:
                logger.error(
                    f"Invalid regex pattern in rule '{rule.name}': {rule.pattern}. "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Invalid regex pattern in rule '{rule.name}': {rule.pattern}. "
                    f"Please fix the pattern configuration. Error: {e}"
                ) from e

        elif rule.mode == ParseMode.KEYWORD:
            # Keyword-based extraction with boundary detection
            start = 0
            iteration_count = 0

            while True:
                # Safety check: prevent infinite loops
                iteration_count += 1
                if iteration_count > MAX_PARSE_ITERATIONS:
                    logger.error(
                        f"Rule '{rule.name}' exceeded maximum iteration limit "
                        f"({MAX_PARSE_ITERATIONS}). Possible infinite loop detected."
                    )
                    raise RuntimeError(
                        f"Parse iteration limit exceeded for rule '{rule.name}'. "
                        f"Check input text or rule configuration."
                    )

                # Safety check: limit number of matches
                if len(results) >= MAX_MATCHES_PER_RULE:
                    logger.warning(
                        f"Rule '{rule.name}' exceeded maximum matches limit "
                        f"({MAX_MATCHES_PER_RULE}). Stopping extraction."
                    )
                    break

                key_pos = text.find(rule.pattern, start)
                if key_pos == -1:
                    break

                # Calculate extraction boundaries
                segment_start = key_pos + len(rule.pattern)
                segment_end = len(text)

                # Use secondary pattern as end boundary if available
                if rule.secondary_pattern:
                    end_pos = text.find(rule.secondary_pattern, segment_start)
                    if end_pos != -1:
                        segment_end = end_pos

                # Extract and clean matched content
                matched_str = text[segment_start:segment_end].strip()
                results.append((key_pos, segment_end, matched_str))

                # Move search window forward
                start = segment_end + 1

                # Safety check: prevent moving backward
                if start <= key_pos:
                    logger.error(
                        f"Rule '{rule.name}' search position not advancing. "
                        f"Stopping to prevent infinite loop."
                    )
                    break

        return results

    def filter_entries(self, parse_result: ParseResult, 
                      filter_type: Literal["all", "successful", "first_n"],
                      n: Optional[int], 
                      rules: List[ParseRule]) -> ParseResult:
        """
        Filter parsing results based on specified criteria.
        
        Args:
            filter_type: Filtering strategy
            n: Required for 'first_n' filter type
            rules: Original rules for validation
            
        Returns:
            Filtered ParseResult
        """
        if filter_type == "all":
            return parse_result
            
        if filter_type == "successful":
            # Only include entries where all rules succeeded
            filtered = []
            for entry in parse_result.entries:
                valid = True
                for rule in rules:
                    fallback = rule.fallback_value or f"missing_{rule.name}"
                    if entry.get(rule.name, "") == fallback:
                        valid = False
                        break
                if valid:
                    filtered.append(entry)
            return ParseResult(entries=filtered)
            
        if filter_type == "first_n" and n is not None:
            return ParseResult(entries=parse_result.entries[:n])
            
        return parse_result