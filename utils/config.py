import getpass
import glob
import os

# This file lives in utils/, so the project folder is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")


def load_env(path=ENV_PATH):
    """Copy KEY=VALUE lines from `path` into os.environ, expanding ~ and $VARS. Existing variables win."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if line.startswith("#") or not sep or not value:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            value = os.path.expanduser(os.path.expandvars(value))
            if "$" in value:
                raise ValueError(f"{key} in {path} uses a variable that isn't set: {value}")
            os.environ.setdefault(key, value)


def _get(name, default):
    return os.environ.get(name) or default


def _dir(name, default):
    """A directory setting. Relative paths are taken from the project folder."""
    return os.path.normpath(os.path.join(ROOT, os.path.expanduser(_get(name, default))))


def _guess_scratch():
    """/scratch/<cluster>/<user> on RCAC clusters. Kernels started from VS Code Remote-SSH don't get $RCAC_SCRATCH."""
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except Exception:
        return None
    found = [d for d in sorted(glob.glob(os.path.join("/scratch", "*", user))) if os.path.isdir(d)]
    return found[0] if found else None


def _find_jdk(*folders):
    """A JDK unpacked by setup_scholar.sh, if JAVA_HOME isn't set."""
    for folder in folders:
        if folder and os.path.exists(os.path.join(folder, "bin", "java")):
            return folder
    return None


if not os.environ.get("RCAC_SCRATCH") and _guess_scratch():
    os.environ["RCAC_SCRATCH"] = _guess_scratch()
load_env()

DIR_SCRATCH = _dir("DIR_SCRATCH", os.environ.get("RCAC_SCRATCH") or "~")
DIR_DATA = _dir("DIR_DATA", "data")
DIR_OUTPUTS = _dir("DIR_OUTPUTS", "outputs")

HF_HOME = os.environ["HF_HOME"] = _dir("HF_HOME", os.path.join(DIR_SCRATCH, "hf_cache"))
PYSERINI_CACHE = os.environ["PYSERINI_CACHE"] = _dir("PYSERINI_CACHE", os.path.join(DIR_SCRATCH, "pyserini_cache"))

JAVA_HOME = os.environ.get("JAVA_HOME") or _find_jdk(os.path.join(DIR_SCRATCH, "jdk-21"))
JAVA_VERSION = int(_get("JAVA_VERSION", 21))

PYSERINI_INDEX = _get("PYSERINI_INDEX", "wikipedia-dpr-100w")

LLAMA_MODEL = _get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLAMA_BACKEND = _get("LLAMA_BACKEND", "hf")  # "hf" (transformers) or "vllm"

# Pyserini builds an OpenAI client on import, and recent `openai` versions raise without a key. Never used here.
os.environ.setdefault("OPENAI_API_KEY", "unused-by-bm25")


def summary():
    names = ["DIR_SCRATCH", "DIR_DATA", "DIR_OUTPUTS", "HF_HOME", "PYSERINI_CACHE", "JAVA_HOME",
             "JAVA_VERSION", "PYSERINI_INDEX", "LLAMA_MODEL", "LLAMA_BACKEND"]
    source = ENV_PATH if os.path.exists(ENV_PATH) else "defaults (no .env)"
    return "\n".join([f"Settings from {source}"] + [f"  {n} = {globals()[n]}" for n in names])