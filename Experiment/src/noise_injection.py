"""Noise generation functions for format robustness testing."""

import re
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import serialize_format_f1, serialize_format_f2, serialize_format_f3, serialize_format_f4, serialize_format_f5


def inject_noise(serialized_text, format_name, noise_level):
    """Inject noise into a serialized format string.
    
    Args:
        serialized_text: The clean serialized format string
        format_name: 'F1', 'F2', 'F3', 'F4', or 'F5'
        noise_level: 1 (minor), 2 (moderate), 3 (major)
    
    Returns:
        Noisy text string
    """
    if noise_level == 0:
        return serialized_text

    lines = serialized_text.split('\n')
    result = list(lines)
    rng = random.Random(42)

    # Find all node IDs
    node_ids = list(set(re.findall(r'\b(v\d+)\b', serialized_text)))

    if noise_level == 1:
        # Minor: swap 2 node IDs
        if len(node_ids) >= 2:
            a, b = node_ids[0], node_ids[1]
            result = [line.replace(a, '___TMP___').replace(b, a).replace('___TMP___', b) for line in result]
        # Shift one capacity value
        caps = re.findall(r'(\d+)', serialized_text)
        if caps:
            pass  # Simple swap is enough for level 1

    elif noise_level == 2:
        # Moderate: swap 3 pairs
        for i in range(min(3, len(node_ids) // 2)):
            if i * 2 + 1 < len(node_ids):
                a, b = node_ids[i * 2], node_ids[i * 2 + 1]
                result = [line.replace(a, '___TMP___').replace(b, a).replace('___TMP___', b) for line in result]

    elif noise_level == 3:
        # Major: omit one edge line, duplicate another
        edge_lines = [i for i, line in enumerate(result) if re.search(r'v\d+.*v\d+', line)]
        if len(edge_lines) >= 2:
            # Remove one
            result.pop(edge_lines[0])
            # Duplicate another (adjust index)
            dup_idx = edge_lines[1] - 1 if edge_lines[1] > edge_lines[0] else edge_lines[1]
            if dup_idx < len(result):
                result.insert(dup_idx + 1, result[dup_idx])

    return '\n'.join(result)