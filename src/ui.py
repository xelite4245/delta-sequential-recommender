"""Interactive CLI user interface"""
from typing import Tuple

DEVIATION_REASONS = {
    "1": "normal",
    "2": "easy",
    "3": "hard",
    "4": "injury",
    "5": "external_stress",
    "6": "other"
}

COMPOUNDS = {
    "1": "squat",
    "2": "bench_press",
    "3": "lat_pulldown",
    "4": "seated_row"
}

def clear_screen():
    """Clear terminal"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text: str):
    """Print formatted header"""
    width = 50
    print("\n" + "="*width)
    print(f"  {text}")
    print("="*width)

def login_screen() -> Tuple[str, str, int]:
    """
    Display login/signup screen
    Returns: (username, password, choice) where choice 1=login, 2=signup, 3=exit
    """
    clear_screen()
    print_header("Personalized Workout Progression")
    
    print("\n1. Login")
    print("2. Sign Up (New Account)")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Invalid selection. Try again.")
    
    if choice == "3":
        return None, None, 3
    
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    return username, password, int(choice)

def compound_menu() -> str:
    """Select which compound to log"""
    print_header("Main Menu")
    
    print("\n📋 Log Session:")
    print("1. Squat")
    print("2. Bench Press")
    print("3. Lat Pulldown")
    print("4. Seated Row")
    print("\n📊 Other:")
    print("5. View Progression Plots")
    print("6. Exit")
    
    while True:
        choice = input("\nSelect (1-6): ").strip()
        if choice in ["1", "2", "3", "4"]:
            return COMPOUNDS[choice]
        elif choice == "5":
            return "view_plots"
        elif choice == "6":
            return None
        print("Invalid selection. Try again.")

def log_session_menu(compound: str) -> Tuple[float, int, float, str]:
    """
    Log session details
    Returns: (weight, reps, rpe, deviation_reason)
    """
    print_header(f"Log Session: {compound.title()}")
    
    # Weight
    while True:
        try:
            weight = float(input("Weight (lbs): ").strip())
            if weight <= 0:
                print("Weight must be positive. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Enter a number.")
    
    # Reps
    while True:
        try:
            reps = int(input("Reps: ").strip())
            if reps <= 0:
                print("Reps must be positive. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Enter a whole number.")
    
    # RPE
    while True:
        try:
            rpe = float(input("RPE (1-10): ").strip())
            if not (1 <= rpe <= 10):
                print("RPE must be between 1 and 10.")
                continue
            break
        except ValueError:
            print("Invalid input. Enter a number.")
    
    # Deviation reason
    print("\nHow did this set feel?")
    print("1. Normal (as expected)")
    print("2. Easy (could have done more)")
    print("3. Hard (struggled more than expected)")
    print("4. Injury/Pain (limited by pain)")
    print("5. External Stress (sleep, stress, etc.)")
    print("6. Other (please add notes)")
    
    deviation_reason = None
    deviation_notes = ""
    
    while True:
        choice = input("\nSelect (1-6): ").strip()
        if choice in DEVIATION_REASONS:
            deviation_reason = DEVIATION_REASONS[choice]
            if choice == "6":
                deviation_notes = input("Brief notes: ").strip()
            break
        print("Invalid selection. Try again.")
    
    return weight, reps, rpe, deviation_reason

def show_recommendation(
    compound: str,
    current_weight: float,
    current_reps: int,
    recommended_weight: float,
    source: str,
    reason: str = None,
    session_count: int = 0
):
    """Display prediction recommendation"""
    print_header(f"Next Session Recommendation")
    
    print(f"\nCompound: {compound.title()}")
    print(f"Current: {current_weight} lbs × {current_reps} reps")
    print(f"Recommended: {recommended_weight:.1f} lbs")
    print(f"Change: {recommended_weight - current_weight:+.1f} lbs")
    
    print(f"\nPrediction Method: {source.upper()}")
    
    if reason:
        print(f"Reason: {reason}")
    
    if session_count <= 15:
        print(f"\n📊 Progress: {session_count} sessions logged")
        if session_count < 15:
            print(f"   Log {15 - session_count} more sessions to enable ML predictions")
        else:
            print("   ML model calibrating...")
    else:
        print(f"\n📊 ML model enabled ({session_count} sessions)")
    
    print("\n" + "="*50)

def continue_menu() -> str:
    """Ask user to log another or exit"""
    while True:
        choice = input("\nLog another session? (y/n): ").strip().lower()
        if choice in ["y", "n"]:
            return choice
        print("Invalid selection. Enter 'y' or 'n'.")

def error_message(message: str):
    """Display error message"""
    print(f"\n❌ Error: {message}")
    input("Press Enter to continue...")

def success_message(message: str):
    """Display success message"""
    print(f"\n✓ {message}")
    input("Press Enter to continue...")

def plots_menu() -> str:
    """
    Display plots viewing menu
    Returns: compound name, or None to go back
    """
    print_header("View Progression Plots")
    
    print("\n1. Squat")
    print("2. Bench Press")
    print("3. Lat Pulldown")
    print("4. Seated Row")
    print("5. Go Back")
    
    while True:
        choice = input("\nSelect (1-5): ").strip()
        if choice in COMPOUNDS or choice == "5":
            if choice == "5":
                return None
            return COMPOUNDS[choice]
        print("Invalid selection. Try again.")
