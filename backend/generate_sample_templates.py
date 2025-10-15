#!/usr/bin/env python3
"""
Generate Sample Check Templates for Testing

This script creates synthetic check templates for template matching testing.
For production, replace these with real bank check templates.

Usage:
    python generate_sample_templates.py
"""

import cv2
import numpy as np
from pathlib import Path


def create_check_template(
    bank_name: str,
    check_type: str,
    width: int = 800,
    height: int = 400,
    seed: int = None
) -> np.ndarray:
    """
    Generate a synthetic check template.
    
    Args:
        bank_name: Name of the bank (e.g., "Wells Fargo")
        check_type: Type of check ("personal" or "business")
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducible patterns
        
    Returns:
        numpy array representing the check image
    """
    if seed:
        np.random.seed(seed)
    
    # Create blank check (light beige background)
    check = np.ones((height, width, 3), dtype=np.uint8) * 250
    check[:, :, 0] = 245  # Slight blue tint
    check[:, :, 1] = 245
    check[:, :, 2] = 250
    
    # Add background pattern (security feature)
    for i in range(0, height, 20):
        for j in range(0, width, 20):
            if (i + j) % 40 == 0:
                cv2.circle(check, (j, i), 2, (230, 230, 235), -1)
    
    # Bank name (top left)
    cv2.putText(
        check,
        bank_name.upper(),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 50, 150),
        3
    )
    
    # Check type indicator
    type_text = f"{check_type.upper()} CHECK"
    cv2.putText(
        check,
        type_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (100, 100, 100),
        1
    )
    
    # Check number (top right)
    check_number = "0001"
    cv2.putText(
        check,
        check_number,
        (width - 100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2
    )
    
    # Date line
    cv2.line(check, (width - 200, 100), (width - 20, 100), (100, 100, 100), 1)
    cv2.putText(
        check,
        "Date",
        (width - 220, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 100, 100),
        1
    )
    
    # Pay to the order of line
    cv2.putText(
        check,
        "PAY TO THE",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    cv2.putText(
        check,
        "ORDER OF",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    cv2.line(check, (130, 170), (width - 120, 170), (0, 0, 0), 1)
    
    # Dollar box
    cv2.rectangle(check, (width - 110, 140), (width - 20, 180), (0, 0, 0), 2)
    cv2.putText(
        check,
        "$",
        (width - 105, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2
    )
    
    # Amount in words line
    cv2.line(check, (20, 220), (width - 120, 220), (0, 0, 0), 1)
    cv2.putText(
        check,
        "Dollars",
        (width - 110, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    
    # Bank address (bottom left)
    cv2.putText(
        check,
        "123 Bank Street",
        (20, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 100, 100),
        1
    )
    cv2.putText(
        check,
        "City, ST 12345",
        (20, height - 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 100, 100),
        1
    )
    
    # Memo line
    cv2.putText(
        check,
        "MEMO",
        (20, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 100, 100),
        1
    )
    cv2.line(check, (70, height - 50), (350, height - 50), (100, 100, 100), 1)
    
    # Signature line
    cv2.line(check, (width - 250, height - 50), (width - 20, height - 50), (0, 0, 0), 1)
    
    # MICR line (bottom) - simulated with special font
    micr_line = "⑆000080413⑆⒜ ⑆053101561⒜⒜ 2079900120668⑆"
    cv2.putText(
        check,
        micr_line,
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )
    
    # Add bank logo (simple placeholder)
    logo_color = (0, 50, 150) if check_type == "personal" else (150, 50, 0)
    cv2.circle(check, (width - 100, 280), 40, logo_color, -1)
    cv2.putText(
        check,
        bank_name[0],
        (width - 115, 295),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )
    
    return check


def generate_all_templates(output_dir: str = "./templates"):
    """Generate sample templates for major banks."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define banks and check types
    banks = [
        ("Wells Fargo", "wells_fargo", 42),
        ("Chase Bank", "chase", 43),
        ("Bank of America", "bank_of_america", 44),
        ("Citibank", "citibank", 45),
        ("US Bank", "usbank", 46),
    ]
    
    check_types = ["personal", "business"]
    
    templates_created = []
    
    print("🏦 Generating Sample Check Templates...")
    print("=" * 60)
    
    for bank_display, bank_name, seed in banks:
        for check_type in check_types:
            # Generate template
            template = create_check_template(
                bank_name=bank_display,
                check_type=check_type,
                seed=seed
            )
            
            # Save to file
            filename = f"{bank_name}_{check_type}.jpg"
            filepath = output_path / filename
            
            cv2.imwrite(str(filepath), template)
            templates_created.append(filename)
            
            print(f"✓ Created: {filename}")
    
    print("=" * 60)
    print(f"✅ Successfully created {len(templates_created)} templates!")
    print(f"📁 Location: {output_path.absolute()}")
    print()
    print("📝 Note: These are SYNTHETIC templates for TESTING ONLY!")
    print("   For production, replace with real bank check templates.")
    print()
    print("🚀 Next steps:")
    print("   1. Review templates in the templates/ directory")
    print("   2. Add real bank templates (see templates/README.md)")
    print("   3. Test with: make start")
    print()
    
    return templates_created


def main():
    """Main entry point."""
    try:
        templates = generate_all_templates()
        
        print("🎯 Template Statistics:")
        print(f"   Total templates: {len(templates)}")
        print(f"   Banks covered: 5 (Wells Fargo, Chase, BoA, Citi, US Bank)")
        print(f"   Check types: 2 (Personal, Business)")
        print()
        print("✨ You're ready to test template matching!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

