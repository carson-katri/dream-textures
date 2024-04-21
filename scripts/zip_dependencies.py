import shutil
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    deps = root / '.python_dependencies'
    deps_to_zip = [deps / 'transformers']

    for dep in deps_to_zip:
        if not dep.exists():
            raise FileNotFoundError(dep)
        elif not dep.is_dir():
            raise EnvironmentError(f"not a directory {dep}")

    zip_deps_path = root / '.python_dependencies.zip'
    zip_deps_path.unlink(True)
    with zipfile.PyZipFile(str(zip_deps_path), mode='x') as zip_deps:
        for dep in deps_to_zip:
            zip_deps.writepy(str(dep))
            shutil.rmtree(str(dep))


if __name__ == '__main__':
    main()
