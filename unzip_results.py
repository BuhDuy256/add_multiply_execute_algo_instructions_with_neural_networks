
import os
import zipfile
from pathlib import Path

source_dir = Path(r"c:\Users\ASUS\Downloads\add_multiply_execute_algo_instructions_with_neural_networks\exp-res")
target_dir = source_dir / "extracted"
target_dir.mkdir(exist_ok=True)

print(f"Extracting zips from {source_dir} to {target_dir}...")

for item in source_dir.glob("*.zip"):
    print(f"Processing {item.name}...")
    try:
        with zipfile.ZipFile(item, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            print(f"  Extracted {item.name}")
    except Exception as e:
        print(f"  Error extracting {item.name}: {e}")

print("Done extraction.")
