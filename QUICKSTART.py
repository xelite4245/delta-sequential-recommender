#!/usr/bin/env python3
"""
QUICK START GUIDE
Run this script once to set up everything, then use: python run_app.py
"""
import sys
from pathlib import Path
import subprocess

repo_root = Path(__file__).parent

print("="*60)
print("PERSONALIZED WORKOUT PROGRESSION - SETUP")
print("="*60)

# Step 1: Initialize auth database
print("\n1. Initializing authentication database...")
try:
    result = subprocess.run([sys.executable, "data/auth/init_auth_db.py"], cwd=repo_root, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✓ {result.stdout.strip()}")
    else:
        print(f"   ✗ Error: {result.stderr}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Step 2: Initialize session audit table
print("\n2. Initializing session audit table...")
try:
    result = subprocess.run([sys.executable, "init_session_audit.py"], cwd=repo_root, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✓ {result.stdout.strip()}")
    else:
        print(f"   ✗ Error: {result.stderr}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Step 3: Verify files exist
print("\n3. Verifying setup...")
required_files = [
    "data/auth/app_users.db",
    "data/user_data.db",
    "run_app.py",
    "src/auth.py",
    "src/ui.py",
    "src/session_logger.py",
    "src/model_quality.py",
    "src/recommendation_engine.py",
]

all_ok = True
for file in required_files:
    path = repo_root / file
    if path.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} NOT FOUND")
        all_ok = False

# Step 4: Run tests
print("\n4. Running validation tests...")
try:
    result = subprocess.run([sys.executable, "test_app_modules.py"], cwd=repo_root, capture_output=True, text=True)
    if "ALL MODULES" in result.stdout:
        print("   ✓ All modules load successfully")
    else:
        print(f"   ⚠ Module test output:\n{result.stdout}")
except Exception as e:
    print(f"   ✗ Test failed: {e}")

# Step 5: Ready!
print("\n" + "="*60)
if all_ok:
    print("✓ SETUP COMPLETE")
    print("="*60)
    print("\nYou're ready to go!")
    print("\nNext steps:")
    print("1. Run: python run_app.py")
    print("2. Login with User2 / password")
    print("3. Log your first session!")
    print("\nOr create a new account:")
    print("1. Run: python run_app.py")
    print("2. Select 'Sign Up'")
    print("3. Create your account")
else:
    print("⚠ SETUP INCOMPLETE")
    print("="*60)
    print("Please check the errors above and try again.")

print("\nFor help, see: APP_README.md")
