#!/usr/bin/env python3
"""
Create a rounded square icon from the glowface.png image
"""

from PIL import Image, ImageDraw
import os

def create_rounded_icon(input_path, output_path, size=512, corner_radius=100):
    """
    Create a rounded square icon from an input image.

    Args:
        input_path: Path to input image
        output_path: Path to save rounded icon
        size: Output size (will be square)
        corner_radius: Radius of corners in pixels
    """
    # Load and resize image
    img = Image.open(input_path).convert("RGBA")
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Create mask for rounded corners
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    # Draw rounded rectangle on mask
    draw.rounded_rectangle(
        [(0, 0), (size, size)],
        radius=corner_radius,
        fill=255
    )

    # Apply mask to image
    output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    output.paste(img, (0, 0))
    output.putalpha(mask)

    # Save the result
    output.save(output_path, 'PNG')
    print(f"Created rounded icon: {output_path}")
    print(f"  Size: {size}x{size}")
    print(f"  Corner radius: {corner_radius}px")

if __name__ == "__main__":
    input_file = "assets/glowface.png"
    output_file = "assets/glowface_rounded.png"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit(1)

    # Create rounded version
    create_rounded_icon(input_file, output_file, size=512, corner_radius=100)
    print(f"\n✓ Rounded icon created successfully!")
