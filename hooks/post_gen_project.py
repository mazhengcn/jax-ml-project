#!/usr/bin/env python

import subprocess
import pathlib


def init_git_repo():
    """
    Initialize a git repository and make an initial commit.
    """
    # Initialize a new git repository
    subprocess.call(["git", "init", "--initial-branch=main"])
    # Add all files to the staging area
    subprocess.call(["git", "add", "*"])
    # Make an initial commit
    subprocess.call(["git", "commit", "-m", "Initial commit"])


if __name__ == "__main__":

    if '{{ cookiecutter.create_author_file }}' != 'y':
        pathlib.Path('AUTHORS.rst').unlink()
        pathlib.Path('docs', 'authors.rst').unlink()

    if 'no' in '{{ cookiecutter.command_line_interface|lower }}':
        pathlib.Path('src', '{{ cookiecutter.module_name }}', 'cli.py').unlink()

    if 'Not open source' == '{{ cookiecutter.open_source_license }}':
        pathlib.Path('LICENSE').unlink()

    init_git_repo()
