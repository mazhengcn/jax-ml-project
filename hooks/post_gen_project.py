import pathlib
import subprocess


if "Not open source" == "{{ cookiecutter.open_source_license }}":
    pathlib.Path("LICENSE").unlink()

subprocess.call(["git", "init", "--initial-branch=main"])
subprocess.call(["git", "add", "*"])
subprocess.call(["git", "commit", "-m", "Initial commit"])
