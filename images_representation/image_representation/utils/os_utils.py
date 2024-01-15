import os
from typing import List


def get_files_paths_under_dir(director_path: str) -> List[str]:
    return [os.path.join(director_path, path) for path in os.listdir(director_path)]
