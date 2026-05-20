from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedFormat:
    value: str
    inferred_from: str | None = None


def parse_format_prefix_arg(
    format_value: str,
    *,
    allowed_formats: Sequence[str],
    label: str,
) -> ResolvedFormat:
    formats = tuple(allowed_formats)
    if format_value in formats:
        return ResolvedFormat(value=format_value)

    matches = tuple(fmt for fmt in formats if fmt.startswith(format_value))
    if len(matches) == 1:
        return ResolvedFormat(value=matches[0], inferred_from=format_value)

    allowed = ", ".join(formats)
    if len(matches) > 1:
        match_list = ", ".join(matches)
        raise argparse.ArgumentTypeError(
            f"{label} prefix {format_value!r} is ambiguous; matches: {match_list}"
        )
    raise argparse.ArgumentTypeError(f"{label} must be one of: {allowed}")
