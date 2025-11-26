import subprocess
import sys
import os

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ {description} - COMMAND NOT FOUND")
        return False

def check_python_version():
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version OK (>= 3.8)")
        return True
    else:
        print("✗ Python version too old (need >= 3.8)")
        return False

def check_system_dependencies():
    print("\n" + "="*60)
    print("CHECKING SYSTEM DEPENDENCIES")
    print("="*60)
    
    results = {}
    
    results['ffmpeg'] = run_command(
        ['ffmpeg', '-version'],
        "Checking ffmpeg"
    )
    
    results['python'] = check_python_version()
    
    print("\n" + "="*60)
    print("SYSTEM DEPENDENCIES SUMMARY")
    print("="*60)
    for dep, status in results.items():
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"{dep}: {status_str}")
    
    return all(results.values())

def install_python_packages():
    print("\n" + "="*60)
    print("INSTALLING PYTHON PACKAGES")
    print("="*60)
    
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"✗ {requirements_file} not found")
        return False
    
    return run_command(
        [sys.executable, '-m', 'pip', 'install', '-r', requirements_file],
        "Installing requirements"
    )

def verify_installation():
    print("\n" + "="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    
    packages = [
        ('yt-dlp', 'yt_dlp'),
        ('whisper', 'whisper'),
        ('pandas', 'pandas'),
        ('yaml', 'yaml'),
        ('tqdm', 'tqdm'),
        ('torch', 'torch'),
    ]
    
    results = {}
    
    for display_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name} - OK")
            results[display_name] = True
        except ImportError:
            print(f"✗ {display_name} - NOT FOUND")
            results[display_name] = False
    
    return all(results.values())

def check_gpu():
    print("\n" + "="*60)
    print("CHECKING GPU AVAILABILITY")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return False

def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "Dominican-eaters Audio Pipeline Setup" + " "*10 + "║")
    print("╚" + "="*58 + "╝")
    
    system_ok = check_system_dependencies()
    
    if not system_ok:
        print("\n" + "="*60)
        print("⚠ WARNING: Some system dependencies are missing")
        print("="*60)
        print("\nPlease install missing dependencies:")
        print("\nUbuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install ffmpeg python3-pip")
        print("\nmacOS:")
        print("  brew install ffmpeg")
        print("\nWindows:")
        print("  Download ffmpeg from: https://ffmpeg.org/download.html")
        print("="*60)
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("\nSetup cancelled.")
            sys.exit(1)
    
    packages_ok = install_python_packages()
    
    if not packages_ok:
        print("\n✗ Failed to install Python packages")
        sys.exit(1)
    
    verify_ok = verify_installation()
    
    if not verify_ok:
        print("\n✗ Installation verification failed")
        sys.exit(1)
    
    check_gpu()
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "✓ SETUP COMPLETED SUCCESSFULLY" + " "*13 + "║")
    print("╚" + "="*58 + "╝")
    print("\nNext steps:")
    print("  1. Review config.yaml")
    print("  2. Run: ./run_pipeline.sh --help")
    print("  3. Start processing: ./run_pipeline.sh --module lyrics-eater")
    print("\n")

if __name__ == '__main__':
    main()
