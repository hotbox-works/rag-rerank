import os
import re
import subprocess

from utils import config

def java_major(java_home):
    try:
        out = subprocess.run([os.path.join(java_home, "bin", "java"), "-version"],
                             capture_output=True, text=True).stderr
    except (FileNotFoundError, PermissionError):
        return 0
    m = re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else 0


def ensure_java(java_home=config.JAVA_HOME, version=config.JAVA_VERSION):
    """Check that JAVA_HOME holds a JDK >= `version`, put it first on PATH, and return its major version."""
    if not java_home:
        raise RuntimeError(f"JAVA_HOME isn't set. Set it in .env to a Java {version}+ JDK.")
    found = java_major(java_home)
    if found < version:
        raise RuntimeError(f"JAVA_HOME={java_home} has Java {found or 'missing'}; Pyserini needs {version}+.")
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ["PATH"]
    return found


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU in this session. Start the Jupyter session on an A40 or A30.")
    return torch.cuda.get_device_name(0)


def setup():
    print(config.summary())
    print("Java", ensure_java())
    print("GPU:", check_gpu())
