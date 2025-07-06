#!/usr/bin/env python3
"""
Voting UI for the RLHF loop.

This script implements a CLI-based interface for the complete RLHF workflow:
1. Generate a prompt
2. Generate completions
3. Display completions to user
4. Collect feedback (votes, confidence, annotations)
5. Log all data for training
"""

import sys
import os
import json
from pathlib import Path
import datetime

# Add the project root to the Python path so imports work correctly
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Now import the modules
from prompts.generator import generate_prompt, log_prompt
from utils.completions import generate_completions

def voting_loop():
    """Run a single iteration of the RLHF voting loop."""
    # Step 1: Generate and log prompt
    prompt_data = generate_prompt()
    log_prompt(prompt_data)
    print("\n🧠 Prompt:")
    print(f"> {prompt_data['generated_prompt']}")

    # Step 2: Generate completions
    result = generate_completions(prompt_data["generated_prompt"], n_completions=2)
    completions = result["completions"]

    # Step 3: Display completions
    print("\n📝 Completions:")
    for i, comp in enumerate(completions):
        # Display first 300 chars with ... if longer
        preview = comp[:300] + ("..." if len(comp) > 300 else "")
        print(f"\n[{i}] {preview}")
    
    # Show full completions if requested
    show_full = input("\nView full completions? (y/n): ").lower().strip()
    if show_full == 'y':
        for i, comp in enumerate(completions):
            print(f"\n--- COMPLETION [{i}] ---\n{comp}\n")

    # Step 4: Get user input
    while True:
        try:
            chosen_index = int(input("\n✅ Enter the number of the preferred completion (0 or 1): "))
            if chosen_index not in [0, 1]:
                raise ValueError("Please enter either 0 or 1")
            break
        except ValueError:
            print("Invalid input. Please enter either 0 or 1.")
    
    while True:
        try:
            confidence = float(input("📊 Enter your confidence in the choice (0.0 - 1.0): "))
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            break
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0.")
    
    annotation = input("✍️ Optional annotation (press Enter to skip): ").strip()

    # Step 5: Construct and log vote
    vote_entry = {
        "prompt": prompt_data["generated_prompt"],
        "completions": completions,
        "chosen_index": chosen_index,
        "confidence": confidence,
        "annotation": annotation if annotation else None,
        "generation_metadata": result,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    log_path = Path(project_root) / "data" / "votes.jsonl"
    # Ensure directory exists
    os.makedirs(log_path.parent, exist_ok=True)
    
    # Append the vote to the log file
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(vote_entry) + "\n")

    print("\n✅ Vote logged successfully.")
    print(f"Chosen completion: [{chosen_index}] with confidence {confidence}")

def main():
    """Main entry point for the voting UI."""
    print("=== RLHF Voting Interface ===")
    print("This tool helps collect human feedback on AI completions.")
    
    continue_voting = True
    votes_count = 0
    
    while continue_voting:
        voting_loop()
        votes_count += 1
        
        choice = input(f"\nVotes collected: {votes_count}. Continue voting? (y/n): ").lower().strip()
        continue_voting = choice == 'y'
    
    print(f"\nThank you! {votes_count} votes collected and logged to data/votes.jsonl")

if __name__ == "__main__":
    main() 