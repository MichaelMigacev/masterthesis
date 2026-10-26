import pickle
import subprocess
import rospkg
import tyro


def get_git_revision_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


def get_git_revision_short_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


def get_short_and_long_git_hashes_for_ros_package(package_name):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path(package_name)
    hash = get_git_revision_short_hash(package_path)
    short_hash = get_git_revision_hash(package_path)
    return short_hash, hash


def dump_config(config, file_path):
    yaml_str = tyro.to_yaml(config)
    with open(file_path+".yaml", "w") as f:
        f.write(yaml_str)


def dump_python_object(o, file_path, protocol=5):
    with open(file_path + ".pickle"+str(protocol), "wb") as f:
        pickle.dump(o, f, protocol=protocol)


def load_python_object(file_path, protocol=5):
    with open(file_path + ".pickle"+str(protocol), "rb") as f:
        o = pickle.load(f)
    return o


def get_conda_environment_string(env_name):
    return subprocess.check_output("conda env export --name "+env_name+" | grep -v \"^prefix: \"", shell=True)
