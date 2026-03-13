import os
import re

replacements = [
    (r'from implementations\.solvers\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.solvers.\1 import'),
    (r'from implementations\.physics\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.physics.\1 import'),
    (r'from implementations\.factories\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.factories.\1 import'),
    (r'from implementations\.file_io\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.io_utils.\1 import'),
    (r'import implementations\.', r'import fast_heat_solv.'),
    (r'from core\.standalone_runner import', r'from fast_heat_solv.solver_base import'),
    (r'from core\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.core.\1 import'),
    (r'import core\.', r'import fast_heat_solv.core.'),
    (r'from interfaces\.([a-zA-Z0-9_]+) import', r'from fast_heat_solv.interfaces.\1 import'),
    (r'import interfaces\.', r'import fast_heat_solv.interfaces.'),
    (r'from utils\.spectral_helpers import', r'from fast_heat_solv.physics.spectral_helpers import'),
    (r'import utils\.spectral_helpers as ([a-zA-Z0-9_]+)', r'from fast_heat_solv.physics import spectral_helpers as \1'),
    (r'from utils\.gcode_path import', r'from fast_heat_solv.io_utils.gcode_path import'),
    (r'from utils\.([a-zA-Z0-9_]+) import', r'from tests.\1 import')     
]

def apply_replacements(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content
    for pattern, repl in replacements:
        new_content = re.sub(pattern, repl, new_content)
    
    # Specific edge case overrides
    new_content = new_content.replace('from research.cut_views import', 'from research.cut_views import')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

for root, _, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root or 'venv' in root or 'env' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            apply_replacements(os.path.join(root, file))

