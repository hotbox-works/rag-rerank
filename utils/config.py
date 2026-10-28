import os

ROOT = os.path.dirname(os.path.abspath(__file__))
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


load_env()

DIR_SCRATCH = _dir("DIR_SCRATCH", os.environ.get("RCAC_SCRATCH") or "~")
DIR_DATA = _dir("DIR_DATA", "data")
DIR_OUTPUTS = _dir("DIR_OUTPUTS", "outputs")

HF_HOME = os.environ["HF_HOME"] = _dir("HF_HOME", os.path.join(DIR_SCRATCH, "hf_cache"))
PYSERINI_CACHE = os.environ["PYSERINI_CACHE"] = _dir("PYSERINI_CACHE", os.path.join(DIR_SCRATCH, "pyserini_cache"))

JAVA_HOME = os.environ.get("JAVA_HOME")
JAVA_VERSION = int(_get("JAVA_VERSION", 21))

PYSERINI_INDEX = _get("PYSERINI_INDEX", "wikipedia-dpr-100w")


def summary():
    names = ["DIR_SCRATCH", "DIR_DATA", "DIR_OUTPUTS", "HF_HOME", "PYSERINI_CACHE", "JAVA_HOME",
             "JAVA_VERSION", "PYSERINI_INDEX"]
    source = ENV_PATH if os.path.exists(ENV_PATH) else "defaults (no .env)"
    return "\n".join([f"Settings from {source}"] + [f"  {n} = {globals()[n]}" for n in names])
