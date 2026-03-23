import json
import sys

try:
    with open(r'C:\Users\Rohan Das\Downloads\SPPULSTM2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open('extracted_code.py', 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                out.write(f"\n# --- Cell {i} ---\n")
                out.write("".join(cell.get('source', [])))
                out.write("\n")
    print("Extraction successful. Saved to extracted_code.py")
except Exception as e:
    print(f"Error: {e}")
