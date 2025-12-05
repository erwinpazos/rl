"""
Check that all required packages are installed.
"""
import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("CHECKING INSTALLATION")
    print("="*60)
    print()
    
    packages = [
        ("gymnasium", "gymnasium"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("mujoco", "mujoco"),
        ("tensorboard", "tensorboard"),
        ("tyro", "tyro"),
        ("matplotlib", "matplotlib"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("="*60)
    
    if all_installed:
        print("ALL PACKAGES INSTALLED ✓")
        print("="*60)
        print("\nYou're ready to start training!")
        print("Run: python test_env.py")
        return 0
    else:
        print("SOME PACKAGES MISSING ✗")
        print("="*60)
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print("  pip install gymnasium torch numpy mujoco tensorboard tyro matplotlib")
        return 1

if __name__ == "__main__":
    sys.exit(main())
