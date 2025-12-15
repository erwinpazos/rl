#!/usr/bin/env python3
"""
Test script to find the basic control model.
"""
import os
import glob

def find_basic_model():
    """Find the basic control model."""
    search_paths = [
        "../basic_control/behavioral_cloning_model.pth",
        "../basic_control/models/basic_robot_control_*/best_model.pth",
        "../basic_control/models/basic_robot_control_*/basic_control.pth", 
        "basic_control/behavioral_cloning_model.pth",
        "basic_control/models/basic_robot_control_*/best_model.pth",
        "basic_control/models/basic_robot_control_*/basic_control.pth"
    ]
    
    print("Current working directory:", os.getcwd())
    print("\nSearching for basic control model...")
    
    for pattern in search_paths:
        print(f"\nTrying pattern: {pattern}")
        files = glob.glob(pattern)
        print(f"  Found files: {files}")
        if files:
            if "*" in pattern:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"  ✓ FOUND: {files[0]}")
            return files[0]
        else:
            print(f"  ✗ No files found")
    
    print("\n❌ No model found!")
    return None

if __name__ == "__main__":
    model_path = find_basic_model()
    if model_path:
        print(f"\n🎉 SUCCESS: Model found at {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        print(f"File size: {os.path.getsize(model_path)} bytes")
    else:
        print("\n💥 FAILED: No model found")