from pathlib import Path
from sys import platform

from subprocess import run, PIPE


def drive(drive_name: str):
    """
    Get drive path for both Windows and WSL (Windows Subsystem for Linux) path.

    For example: drive('c')
    If platform is windows, it will return:
        'C:'

    IF platform is WSL, it will return:
        '/mnt/c'

    :param drive_name: name of drive e.g. 'c' or 'd'
    :return: pathlib.Path
    """
    if platform == 'win32':
        return Path(drive_name + ':/')
    elif platform == 'linux':
        return Path('/mnt', drive_name.lower())

    raise NotImplementedError(f'{platform} is not support yet.')


# %%

def wsl_path(path: str, reverse=False):
    """
    Convert path between Windows path and WSL path
    using 'wslpath' command execute in terminal and get output
    as string.
    see: https://devblogs.microsoft.com/commandline/windows10v1803/

    :param reverse: If True, convert WSL path to Windows path
    :param path: path to be convert
    :return: converted path
    """

    arg = '-m' if reverse else '-u'
    re = run(['wslpath', arg, path], stdout=PIPE)
    return re.stdout.strip().decode('utf-8')


def wsl_paths(paths, reverse=False):
    """
    Convert path between Windows path and WSL path as whole
    :param paths: list contains path
    :param reverse: If True, convert WSL path to Windows path
    :return: converted paths as list
    """
    return [wsl_path(str(p), reverse=reverse) for p in paths]

