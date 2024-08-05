# Installation guide

## CUDA ToolKit 10.1 (Optional)

[Original PRNet](https://github.com/YadiraF/PRNet) uses Tensorflow 1, so you have to install CUDA ToolKit if you want to use GPU for PRNet.
If you use CPU, then this step can be skipped.

## Install conda using installer

1. First, go to the Anaconda website at https://www.anaconda.com/products/individual and download the installer for your operating system.

1. Run the installer and follow the instructions to complete the installation. During the installation process, you can choose to add Anaconda to your system path, which will make it easier to access from the command line.

1. Once Anaconda is installed, open a terminal or command prompt and type `conda --version` to verify that it is installed correctly.

## Install conda using terminal in Ubuntu

To install conda only for one user on Ubuntu, you can follow these steps:

1.  Download the Miniconda installation script from the conda website:

        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    This will download the latest version of Miniconda for 64-bit Linux.

1.  Run the installation script with the `-b` flag to install Miniconda in silent mode and the `-p` flag to specify the installation directory:

        bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/<username>/miniconda

    Replace `<username>` with the name of the user for whom you want to install conda. This will install Miniconda in the `/home/<username>/miniconda` directory.

1.  Add the conda binary directory to the user's PATH environment variable:

        echo 'export PATH="/home/<username>/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc

    This will add the `/home/<username>/miniconda/bin` directory to the beginning of the user's PATH environment variable, ensuring that the conda command is used instead of any system-installed version of conda.

1.  Once Anaconda is installed, open a terminal or command prompt and type `conda --version` to verify that it is installed correctly.

## Setting up conda virtual environment

1.  To create a new virtual environment, you can use the conda create command followed by the name of the environment and any packages you want to install. For example, to create a new environment called `rppg` with Python 3.7, you can use the following command:

        conda create --name rppg python=3.7

    This will create a new virtual environment called rppg with Python 3.7 installed

1.  Initialize bash

        conda init bash

1.  To activate the virtual environment, you can use the following command:

        conda activate rppg

1.  Once the environment is activated, you can install packages using `conda install`, just as you would in a regular Python environment. First we have to add conda-forge channels for some packages.

        conda config --add channels conda-forge
        conda config --add channels intel
        conda install --file condarequirements.txt
        conda install -c defaults tensorflow-gpu=1.14.0

1.  Run `conda list` to make sure you have installed all required packages.

**Note**: **tensorflow-gpu 1.14** must be installed from _defaults_ channel if you want to use GPU, hence it is excluded from the requirements file.

## (Not recommended, pip has broken packages) Setup using Python virtualenv & pip

-   Python 3.5-3.7
-   pip 19.0 or later

https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv

    // If Python>=3.3, the venv module is already in the Python standard library, so this step could be skipped.
    python3 -m pip install --user virtualenv

    // Create
    python3 -m venv <path to env>

    // Activate
    source <path to env>/bin/activate

    // Deactivate
    deactivate

### Install Python packages dependency

    cd <path to repo>
    pip install -r requirements.txt

### Run testing files

    python export_bbox_colormap.py list_test.json




###Use Docker to create a virtual enviroment

To ensure accurate reproduction of the same program, it is essential to establish an identical environment to our previous setup. Below are detailed steps to create the required virtual environment using Docker.

## 1. Create a Docker Image

    First, create a Docker image that includes all necessary dependencies and configurations to ensure smooth operation of the application.

    docker build -t rppg_project_env .


2.With the image ready, proceed to create a container based on the image
    docker run -it --rm -v "${PWD}:/usr/src/app" rppg_project_env


3.Once the container is running, you will be in the terminal interface of the container