"""Create HTML gallery of all figures."""

import os
import sys
import argparse
from pathlib import Path


def generate_preview(figures_dir, output_path):
    """Generate an HTML page showing all figures in a grid."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Graph Sensitivity - Figure Gallery</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }
        .figure { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .figure img { width: 100%; height: auto; border-radius: 4px; }
        .figure h3 { margin: 10px 0 5px; color: #555; }
        .figure p { color: #777; font-size: 14px; margin: 0; }
    </style>
</head>
<body>
    <h1>LLM Graph Representation Sensitivity - Figure Gallery</h1>
    <div class="gallery">
"""

    figures_path = Path(figures_dir)
    if figures_path.exists():
        for fig_file in sorted(figures_path.glob('*.png')):
            rel_path = fig_file.name
            html += f"""
        <div class="figure">
            <img src="{figures_dir}/{rel_path}" alt="{fig_file.stem}">
            <h3>{fig_file.stem.replace('_', ' ').title()}</h3>
            <p>Figure: {fig_file.stem}</p>
        </div>
"""

    html += """
    </div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Gallery saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML figure gallery')
    parser.add_argument('--input', type=str, default='results/comparison/figures')
    parser.add_argument('--output', type=str, default='results/figure_preview.html')
    args = parser.parse_args()
    generate_preview(args.input, args.output)


if __name__ == '__main__':
    main()