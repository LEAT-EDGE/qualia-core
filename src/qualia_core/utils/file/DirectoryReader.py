import os
import glob
from pathlib import Path

class DirectoryReader:
    def read(self, directory: Path=None, ext: str='', recursive: bool=False):
        """
        Return the list of files in a given directory, optionally matching a given extension.

        Remark: case-sensitive.

        :param directory: directory to search in, if empty search is performed in working directory
        :param ext: optional file extension to match, '.' has to be inserted by the caller as required
        :param recursive: do a recursive search
        """
        directory = directory.expanduser()
        return directory.glob(f'{"**/" if recursive else ""}*{ext}')
