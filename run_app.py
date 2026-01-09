#!/usr/bin/env python3
"""Main application entry point"""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from src import auth, ui, session_logger, recommendation_engine, model_quality
from src.plot_generator import generate_and_save_plots, open_plot

def initialize_databases():
    """Initialize auth database on startup"""
    try:
        from data.auth.init_auth_db import init_auth_db
        init_auth_db()
    except Exception as e:
        print(f"Warning: Could not initialize auth database: {e}")

def main():
    """Main application loop"""
    initialize_databases()
    
    while True:
        # Login/Signup screen
        username, password, choice = ui.login_screen()
        
        if choice == 3:  # Exit
            print("\nGoodbye!")
            break
        
        try:
            if choice == 1:  # Login
                user_id, user_data_path = auth.login(username, password)
                print(f"\n✓ Welcome back, {username}!")
            else:  # Sign up
                user_id, user_data_path = auth.register(username, password)
                print(f"\n✓ Account created! Welcome, {username}!")
            
            input("Press Enter to continue...")
            
            # Main menu loop (log sessions)
            while True:
                compound = ui.compound_menu()
                
                if compound is None:  # Exit
                    break
                
                # Check if user wants to view plots instead
                if compound == "view_plots":
                    # Plots menu
                    selected_compound = ui.plots_menu()
                    if selected_compound is not None:
                        plot_path = Path(user_data_path) / "plots" / f"{selected_compound}_progression.png"
                        if plot_path.exists():
                            print(f"\n📊 Opening {selected_compound} progression plot...")
                            if open_plot(plot_path):
                                print("✓ Plot opened in default viewer")
                            else:
                                ui.error_message(f"Could not open plot at {plot_path}")
                        else:
                            ui.error_message(f"No plot found yet. Log a {selected_compound} session first.")
                    continue
                
                # Compute accuracy for any pending predictions from previous sessions
                session_logger.compute_accuracy_for_pending_predictions(user_id, user_data_path)
                
                # Get current session count and last session
                session_count = session_logger.get_session_count(user_data_path, compound)
                last_session = session_logger.get_last_session(user_data_path, compound)
                
                # Log new session
                weight, reps, rpe, deviation_reason = ui.log_session_menu(compound)
                
                # Determine recommendation based on CURRENT history (before this session is logged)
                if last_session is None:
                    # First session for this compound
                    recommended_weight = weight  # No recommendation for first session
                    prediction_source = 'rule_based'
                    reason = 'First session - log more to enable predictions'
                else:
                    # Get last session info
                    last_weight, last_reps, last_rpe = last_session
                    
                    # Get recommendation (based on current sessions)
                    recommended_weight, prediction_source, reason = recommendation_engine.get_recommendation(
                        user_id=user_id,
                        user_data_path=user_data_path,
                        compound=compound,
                        last_weight=last_weight,
                        last_reps=last_reps,
                        last_rpe=last_rpe,
                        session_count=session_count
                    )
                
                # Log the session
                session_logger.log_session(
                    user_id=user_id,
                    user_data_path=user_data_path,
                    compound=compound,
                    weight=weight,
                    reps=reps,
                    rpe=rpe,
                    deviation_reason=deviation_reason,
                    recommended_weight=recommended_weight if recommended_weight else weight,
                    prediction_source=prediction_source
                )
                
                ui.success_message("Session logged!")
                
                # Show recommendation
                if recommended_weight is not None:
                    ui.show_recommendation(
                        compound=compound,
                        current_weight=weight,
                        current_reps=reps,
                        recommended_weight=recommended_weight,
                        source=prediction_source,
                        reason=reason if 'reason' in locals() else None,
                        session_count=session_count + 1
                    )
                else:
                    ui.show_recommendation(
                        compound=compound,
                        current_weight=weight,
                        current_reps=reps,
                        recommended_weight=weight + 2.5,  # Default +2.5 lbs
                        source='rule_based',
                        reason='Log your first session!',
                        session_count=1
                    )
                
                # REGENERATE PLOTS FOR THIS COMPOUND (session just ended)
                print("\n📊 Updating progression plot...")
                plot_path = generate_and_save_plots(user_data_path, compound)
                if plot_path:
                    print(f"✓ Plot saved to {plot_path}")
                else:
                    print("⚠ Could not generate plot")
                
                # Ask if user wants to log another
                if ui.continue_menu() == 'n':
                    break
        
        except auth.AuthError as e:
            ui.error_message(str(e))
        except Exception as e:
            ui.error_message(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
