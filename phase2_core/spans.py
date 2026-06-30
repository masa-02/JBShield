from collections.abc import Mapping


class SpanMappingError(ValueError):
    pass


class AmbiguousSpanError(SpanMappingError):
    pass


def _as_list(ids):
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def find_subsequence_matches(input_ids, target_ids):
    source = _as_list(input_ids)
    target = _as_list(target_ids)
    if not target or len(target) > len(source):
        return []
    matches = []
    width = len(target)
    for idx in range(0, len(source) - width + 1):
        if source[idx : idx + width] == target:
            matches.append((idx, idx + width))
    return matches


def _tokenize_text_variants(tokenizer, text):
    variants = []
    for add_special_tokens in (False, True):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=None,
            )
            ids = encoded["input_ids"] if "input_ids" in encoded else encoded
            ids = _as_list(ids)
            if ids and ids not in variants:
                variants.append(ids)
        except TypeError:
            continue
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = _as_list(ids)
        if ids and ids not in variants:
            variants.append(ids)
    except AttributeError:
        pass
    return variants


def resolve_user_prompt_span(input_ids, tokenizer, prompt, strict=False):
    """Resolve the user prompt content span inside tokenized chat input.

    Official JBShield prompts do not carry explicit span annotations. Phase2 v1
    therefore uses token subsequence matching when possible and falls back to
    the full input sequence only when strict mode is disabled.
    """
    source_ids = _as_list(input_ids)
    if not source_ids:
        if strict:
            raise SpanMappingError("Cannot resolve span for empty token sequence")
        return {
            "start": 0,
            "end": 0,
            "source": "empty_sequence",
            "token_count": 0,
            "sequence_length": 0,
        }

    all_matches = []
    for candidate in _tokenize_text_variants(tokenizer, prompt):
        all_matches.extend(find_subsequence_matches(source_ids, candidate))

    unique_matches = []
    for match in all_matches:
        if match not in unique_matches:
            unique_matches.append(match)

    if len(unique_matches) == 1:
        start, end = unique_matches[0]
        return {
            "start": start,
            "end": end,
            "source": "subsequence",
            "token_count": end - start,
            "sequence_length": len(source_ids),
        }

    if len(unique_matches) > 1:
        if strict:
            raise AmbiguousSpanError(
                f"Prompt span is ambiguous: {len(unique_matches)} matches"
            )
        start, end = unique_matches[0]
        return {
            "start": start,
            "end": end,
            "source": "subsequence_ambiguous_first",
            "token_count": end - start,
            "sequence_length": len(source_ids),
        }

    if strict:
        raise SpanMappingError("Prompt text was not found in tokenized input")
    return {
        "start": 0,
        "end": len(source_ids),
        "source": "whole_sequence_fallback",
        "token_count": len(source_ids),
        "sequence_length": len(source_ids),
    }
