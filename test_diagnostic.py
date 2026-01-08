"""
Automated test runner for dev_diagnostic
Sends menu inputs and captures output
"""

import subprocess
import sys

# Run the diagnostic with predefined inputs: test RFR, test LR, test XG, show results, exit
inputs = "1\n2\n3\n4\n5\n"

proc = subprocess.Popen(
    [sys.executable, "-m", "src.models.dev_diagnostic"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd=r"c:\Users\rezas\GitHub\Personalized-Workout-Progression-System"
)

output, _ = proc.communicate(input=inputs, timeout=300)

# Write to output file
with open(r"c:\Users\rezas\GitHub\Personalized-Workout-Progression-System\output.txt", "w") as f:
    f.write(output)

print("Output saved to output.txt")
print("\n" + "="*80)
print("FIRST 2000 CHARS OF OUTPUT:")
print("="*80)
print(output[:2000])
