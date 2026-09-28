import sys
import os
import platform
import shutil
import importlib.util

print("=== OPERATING SYSTEM & HARDWARE ===")
print(f"OS: {platform.system()} {platform.release()} (Version: {platform.version()})")
print(f"Platform Architecture: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {platform.python_version()} ({sys.version.splitlines()[0]})")

# CPU & Memory info
try:
    import psutil
    cpu_count_phys = psutil.cpu_count(logical=False)
    cpu_count_log = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    print(f"CPU Physical Cores: {cpu_count_phys}")
    print(f"CPU Logical Threads: {cpu_count_log}")
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB ({mem.percent}% used)")
except ImportError:
    cpu_count = os.cpu_count()
    print(f"CPU Cores (os.cpu_count): {cpu_count}")
    # Windows wmic fallback for memory if psutil isn't installed
    try:
        import subprocess
        res = subprocess.check_output("wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value", shell=True).decode()
        print("Memory (wmic):")
        print(res.strip())
    except Exception as e:
        print(f"Memory check fallback error: {e}")

# Disk space for current drive and C drive
for drive in [os.path.splitdrive(os.path.abspath("."))[0] or "C:", "C:"]:
    try:
        total, used, free = shutil.disk_usage(drive + "\\")
        print(f"Disk Space ({drive}): Total: {total / (1024**3):.2f} GB | Used: {used / (1024**3):.2f} GB | Free: {free / (1024**3):.2f} GB ({free/total*100:.1f}% free)")
    except Exception as e:
        print(f"Disk check error for {drive}: {e}")

print("\n=== PYTORCH & CUDA STATUS ===")
if importlib.util.find_spec("torch"):
    import torch
    print(f"PyTorch: Installed (version: {torch.__version__})")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
    else:
        print("Computation Device: CPU only (as constrained)")
else:
    print("PyTorch: NOT installed")

print("\n=== ML, COMPUTER VISION & 3D PACKAGES ===")
packages = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("opencv-python", "cv2"),
    ("Pillow", "PIL"),
    ("ultralytics", "ultralytics"),
    ("matplotlib", "matplotlib"),
    ("open3d", "open3d"),
    ("laspy", "laspy"),
    ("plyfile", "plyfile"),
    ("trimesh", "trimesh"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("timm", "timm"),
    ("numba", "numba"),
    ("tqdm", "tqdm"),
    ("pyyaml", "yaml"),
]

for label, mod_name in packages:
    spec = importlib.util.find_spec(mod_name)
    if spec is not None:
        try:
            m = __import__(mod_name)
            v = getattr(m, "__version__", "installed")
            print(f"- {label:<16}: {v}")
        except Exception as err:
            print(f"- {label:<16}: installed (import error: {err})")
    else:
        print(f"- {label:<16}: NOT installed")

print("\n=== VIRTUAL ENVIRONMENTS DETECTED ===")
for venv_path in [".venv", "venv"]:
    if os.path.isdir(venv_path):
        py_exe = os.path.join(venv_path, "Scripts", "python.exe")
        exists = os.path.isfile(py_exe)
        print(f"- {venv_path}: directory exists, python.exe present: {exists}")
