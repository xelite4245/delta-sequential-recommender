"""
Automated test runner for dev_diagnostic
Sends menu inputs and captures output
"""

import subprocess
import sys
from pathlib import Path

# Get repo root (parent of this file's directory)
repo_root = Path(__file__).parent.parent.parent

# Run the diagnostic with predefined inputs: test RFR, test LR, test XG, show results, exit
inputs = "1\n2\n3\n4\n5\n"

proc = subprocess.Popen(
    [sys.executable, "-m", "src.models.dev_diagnostic"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=str(repo_root)
)

output, _ = proc.communicate(input=inputs, timeout=300)

# Write to output file
output_path = repo_root / "output.txt"
with open(output_path, "w") as f:
    f.write(output)

print("Output saved to output.txt")
print("\n" + "="*80)
print("FIRST 2000 CHARS OF OUTPUT:")
print("="*80)
print(output[:2000])
