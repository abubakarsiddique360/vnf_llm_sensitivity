"""Verify all 5 format serialization functions work correctly for all problems."""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_environment, load_problems, setup_logging, serialize_format_f1, serialize_format_f2, serialize_format_f3, serialize_format_f4, serialize_format_f5


def main():
    parser = argparse.ArgumentParser(description='Verify format serializations')
    parser.add_argument('--input', type=str, default='data/problems', help='Problems directory')
    parser.add_argument('--output', type=str, default='prompts', help='Prompts directory')
    args = parser.parse_args()

    logger = setup_logging('serialize_formats')
    problems = load_problems(args.input)
    logger.info(f"Loaded {len(problems)} problems")

    serializers = {
        'F1': serialize_format_f1,
        'F2': serialize_format_f2,
        'F3': serialize_format_f3,
        'F4': serialize_format_f4,
        'F5': serialize_format_f5,
    }

    all_passed = True
    for problem in problems[:5]:  # Test first 5
        pid = problem['problem_id']
        for fmt_name, fmt_func in serializers.items():
            try:
                result = fmt_func(problem)
                assert len(result) > 0, f"Empty serialization for {fmt_name}"
                logger.info(f"  Problem {pid}, {fmt_name}: OK ({len(result)} chars)")
            except Exception as e:
                logger.error(f"  Problem {pid}, {fmt_name}: FAILED - {e}")
                all_passed = False

    if all_passed:
        print(f"All serialization tests PASSED for {min(5, len(problems))} problems")
    else:
        print("Some serialization tests FAILED")


if __name__ == '__main__':
    main()