import argparse
import subprocess
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=lambda p: Path(p).absolute())
parser.add_argument("--output", type=lambda p: Path(p).absolute())
parser.add_argument("--no-deps", action="store_true")
parser.add_argument("--install", action="store_true")

def main():
    args = parser.parse_args()

    # Copy the files into the packaged addon
    shutil.copytree(args.backend, args.output / args.backend.name, dirs_exist_ok=True)    

    if args.install:
        # Install the dependencies into the package.
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "-r", args.backend / "requirements.txt",
                "--upgrade",
                "--no-cache-dir",
                "--target", args.output / args.backend.name / ".python_dependencies",
            ] + (["--no-deps"] if args.no_deps else []),
            check=True,
            cwd=args.output
        )

if __name__ == '__main__':
    main()