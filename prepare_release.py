import os
import shutil

def increment_version():
    version_file = os.path.join(
        os.path.dirname(__file__),
        'isoCycle',
        'version.py'
    )
    with open(version_file, 'r') as f:
        version = f.read().split('=')[1].strip().strip("'")

    version_parts = [int(x) for x in version.split('.')]
    version_parts[-1] += 1  # increment the patch version
    new_version = '.'.join(str(x) for x in version_parts)

    with open(version_file, 'w') as f:
        f.write(f"__version__ = '{new_version}'")

def clean_dist_directory():
    dist_dir = os.path.join(os.path.dirname(__file__), 'dist')
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)

if __name__ == '__main__':
    clean_dist_directory()
    increment_version()