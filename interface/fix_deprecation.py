#!/usr/bin/env python3
import re
import os
import shutil
from pathlib import Path

def fix_deprecated_api_calls(file_path):
    """
    Fix deprecated Streamlit API calls in the given file.
    - Replace st.experimental_rerun() with st.rerun()
    - Replace use_column_width=True with use_container_width=True
    """
    print(f"Processing file: {file_path}")
    
    # Create a backup of the original file
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at: {backup_path}")
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace experimental_rerun() with rerun()
    pattern1 = r'st\.experimental_rerun\(\)'
    replacement1 = 'st.rerun()'
    original_content = content
    content = re.sub(pattern1, replacement1, content)
    count1 = original_content.count('st.experimental_rerun()') 
    print(f"Replaced {count1} occurrences of st.experimental_rerun()")
    
    # Replace use_column_width with use_container_width
    pattern2 = r'use_column_width\s*=\s*True'
    replacement2 = 'use_container_width=True'
    original_content = content
    content = re.sub(pattern2, replacement2, content)
    count2 = len(re.findall(pattern2, original_content))
    print(f"Replaced {count2} occurrences of use_column_width=True")
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"File updated successfully.")
    return count1 + count2

if __name__ == "__main__":
    # Get the current directory (where this script is located)
    script_dir = Path(__file__).parent
    
    # Target file path
    target_file = script_dir / "attunement_dashboard.py"
    
    if not target_file.exists():
        print(f"Error: Target file {target_file} not found.")
        exit(1)
    
    # Fix the file
    changes_made = fix_deprecated_api_calls(str(target_file))
    
    if changes_made > 0:
        print(f"Total {changes_made} changes made to fix deprecated API calls.")
        print("Please run the dashboard to verify it works correctly.")
    else:
        print("No deprecated API calls were found in the file.") 