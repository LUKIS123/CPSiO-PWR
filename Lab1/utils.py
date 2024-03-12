from os import listdir
from os.path import isfile, join


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def list_files_in_directory(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]
