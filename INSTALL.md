# AWRA Community Modelling System


## Installation

### Python Environment Dependencies

The modelling system has been developed, tested and used in Linux environments. Installation guidance is provided for several flavours of Linux (Ubuntu, Fedora, Mint and “Bash on Ubuntu on Windows”). Guidance is also provided for installing the system on Mac OS X although this platform is not supported.

The modelling system is designed to be run from Python 3 (specifically has been developed and tested with v3.4.2 and tested with 3.5.2) and relies on the following capabilities

* C compiler to build the core model code
* NetCDF library (version 4.3, NOTE: 4.4 creates files that are unreadable with h5py) and Python bindings (NetCDF4)
* HDF5 library (Tested on 1.8 and up) and Python bindings (h5py)
* Python numpy and pandas packages
* IPython/Jupyter notebook for the recommended interactive use of the modelling system
* and Python packages via `pip install`:
  * `matplotlib` for image and graph display
  * `nose` for running tests
  * `cffi` for building the model Python bindings
  * `pyzmq` for inter-process communication

Certain functions also rely on the osgeo/GDAL libraries and corresponding Python bindings. To use shape files GDAL needs to be installed. However, this is optional, as the simulation package will work without it.  GDAL installation should be performed after building the python virtual environment.

For Red Hat flavours of Linux:
* install GDAL with sudo dnf install gdal gdal-devel
* install GDAL Python package with pip install gdal

For Debian flavours of Linux:
* install GDAL with sudo apt-get install gdal-bin libgdal-dev
* install GDAL Python package with pip install gdal

There is always a number of ways to install these pre-requisites on any given operating system and the situation will be different on different OS versions. The following sections give guidance for setting up the pre-requisites on some different operating systems.

#### Linux: Fedora 24
tested on 19/09/2016

virtual environment installation

```
export PYENV="/home/[user]/.venv"
export BUILD="$PYENV/build"

mkdir -p $BUILD
cd $BUILD

### building Python requires several libraries - this will install these libraries system-wide
sudo dnf install openssl-devel bzip2-devel freetype-devel sqlite3-devel
sudo dnf install gcc-c++

### download and install python
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar zxf Python-3.5.2.tgz
cd Python-3.5.2
./configure --prefix=$PYENV
make
make install
cd ..

### create and activate virtual environment
$PYENV/bin/pyvenv $PYENV/virtualenv
. $PYENV/virtualenv/bin/activate

export LD_LIBRARY_PATH="$PYENV/lib:$LD_LIBRARY_PATH"
export CPPFLAGS="$CPPFLAGS -I$PYENV/include -I$PYENV/lib"
export LDFLAGS="$LDFLAGS -L$PYENV/lib"

### download and install latest HDF5(1.8.17)
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz
tar xzf hdf5-1.8.17.tar.gz
cd hdf5-1.8.17
./configure --prefix=$PYENV
make check
make install
cd ..

### download and install NETCDF (do not use latest(4.4.*) since files created cannot be read by h5py)
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-4.3.2.tar.gz
tar xzf netcdf-4.3.2.tar.gz
cd netcdf-4.3.2
./configure --prefix=$PYENV --enable-dap
make check
make install
cd ..

### install python package netCDF4
export USE_SETUPCFG=0
export HDF5_INCDIR=$PYENV/include
export HDF5_LIBDIR=$PYENV/lib
pip install netCDF4==1.2.4

### required python packages
pip install h5py
pip install pyzmq
pip install pandas==0.16.1
pip install matplotlib==1.4.3
pip install ipython[notebook]

### python package cffi and required libraries
sudo dnf install python3-cffi
sudo dnf install libffi-devel
pip install cffi==1.1.2
```


#### Linux: Ubuntu 16.04 LTS
tested on 19/09/2016

virtual environment installation

```
# choose a path to setup virtual env
export PYENV="/home/[user]/.venv"
export BUILD="$PYENV/build"

mkdir -p $BUILD
cd $BUILD

### build requires several libraries - this will install these libraries system-wide
sudo apt-get install libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev
sudo apt-get install libfreetype6-dev libpng-dev

### download and install python
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar zxf Python-3.5.2.tgz
cd Python-3.5.2
./configure --prefix=$PYENV
make
make install
cd ..

### create and activate virtual environment
$PYENV/bin/pyvenv $PYENV/virtualenv
. $PYENV/virtualenv/bin/activate

export LD_LIBRARY_PATH="$PYENV/lib:$LD_LIBRARY_PATH"
export CPPFLAGS="$CPPFLAGS -I$PYENV/include"
export LDFLAGS="$LDFLAGS -L$PYENV/lib"

### download and install latest HDF5(1.8.17)
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz
tar xzf hdf5-1.8.17.tar.gz
cd hdf5-1.8.17
./configure --prefix=$PYENV
make check
make install
cd ..

### download and install NETCDF (do not use latest(4.4.*) since files created cannot be read by h5py)
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-4.3.2.tar.gz
tar xzf netcdf-4.3.2.tar.gz
cd netcdf-4.3.2
./configure --prefix=$PYENV --enable-dap
make check
make install
cd ..

### install python packages
pip install numpy==1.9.3

### download and build python package netCDF4
wget https://pypi.python.org/packages/source/n/netCDF4/netCDF4-1.2.4.tar.gz
tar zxf netCDF4-1.2.4.tar.gz
cd netCDF4-1.2.4
cp setup.cfg.template setup.cfg
### if build cannot find hdf5 then need to export these 2 environment variables
export HDF5_DIR=$PYENV
export NETCDF4_DIR=$PYENV
python setup.py build
python setup.py install
cd ..

### required python packages
pip install h5py
pip install pyzmq
pip install pandas==0.16.1
pip install matplotlib==1.4.3
pip install ipython[notebook]

### python package cffi and required libraries
sudo apt-get install python3-cffi
sudo apt-get install libffi-dev
pip install cffi==1.1.2
```

To read and use shape files GDAL needs to be installed, this is optional, the modelling system will work without it.
* install GDAL with `sudo apt-get install gdal-bin libgdal-dev`
* install GDAL Python package with `pip install gdal`


#### Linux: Mint 18 'Sarah' cinnamon desktop 64-bit
tested on 28/09/2016

downloaded from https://www.linuxmint.com/download.php

virtual environment installation

```
export PYENV="/home/[user]/.venv"
mkdir -p "$PYENV/build"
cd $PYENV/build

### building Python requires several libraries - this will install these libraries system-wide
sudo apt-get install build-essential
sudo apt-get install libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev
sudo apt-get install libfreetype6-dev libpng-dev

# download and install Python
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar zxf Python-3.5.3.tgz
cd Python-3.5.2
./configure --prefix=$PYENV
make
make install
cd ..

### create and activate virtual environment
$PYENV/bin/pyvenv $PYENV/virtualenv
. $PYENV/virtualenv/bin/activate

export LD_LIBRARY_PATH="$PYENV/lib:$LD_LIBRARY_PATH"
export CPPFLAGS="$CPPFLAGS -I$PYENV/include"
export LDFLAGS="$LDFLAGS -L$PYENV/lib"

### download and install latest HDF5(1.8.17)
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz
tar xzf hdf5-1.8.17.tar.gz
cd hdf5-1.8.17
./configure --prefix=$PYENV
make check
make install
cd ..

### download and install NETCDF (do not use latest(4.4.*) since files created cannot be read by h5py)
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-4.3.2.tar.gz
tar xzf netcdf-4.3.2.tar.gz
cd netcdf-4.3.2
./configure --prefix=$PYENV --enable-dap
make check
make install
cd ..

### install python packages
pip install numpy==1.9.3

### install netCDF4 python package
export HDF5_DIR=$PYENV
export NETCDF4_DIR=$PYENV
pip install netCDF4==1.2.4

### python package cffi and required libraries
sudo apt-get install python3-cffi
sudo apt-get install libffi-dev
pip install cffi==1.1.2

### required python packages
pip install h5py
pip install pyzmq
pip install pandas==0.16.1
pip install matplotlib
pip install ipython[notebook]
```


#### Linux: Bash on Ubuntu on Windows with the Windows SubSystem for Linux (WSL)
tested on 28/09/2016

WSL will run on a 64-bit version of Windows 10 Anniversary Update build 14393 or later

To enable WSL, follow the instructions at:  [https://msdn.microsoft.com/en-au/commandline/wsl/install_guide](https://msdn.microsoft.com/en-au/commandline/wsl/install_guide)

Then open a command prompt and type *bash*

Recommend setting up virtual environment on bash file system as this gives control over file permissions with chmod

```
# choose a path to setup virtual env
export PYENV="/home/[user]/.venv"
mkdir -p "$PYENV/build"
cd $PYENV/build

### build requires several libraries - this will install these libraries system-wide
sudo apt-get install build-essential
sudo apt-get install m4
sudo apt-get install libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev
sudo apt-get install libfreetype6-dev libpng-dev

# download and install Python
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz
tar zxf Python-3.5.2.tgz
cd Python-3.5.2
./configure --prefix=$PYENV
make
make install
cd ..

### create and activate virtual environment
$PYENV/bin/pyvenv $PYENV/virtualenv
. $PYENV/virtualenv/bin/activate

export LD_LIBRARY_PATH="$PYENV/lib:$LD_LIBRARY_PATH"
export CPPFLAGS="$CPPFLAGS -I$PYENV/include"
export LDFLAGS="$LDFLAGS -L$PYENV/lib"

### download and install latest HDF5(1.8.17)
wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz
tar xzf hdf5-1.8.17.tar.gz
cd hdf5-1.8.17
./configure --prefix=$PYENV
make check
make install
cd ..

### download and install NETCDF (do not use latest(4.4.*) since files created cannot be read by h5py)
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/old/netcdf-4.3.2.tar.gz
tar xzf netcdf-4.3.2.tar.gz
cd netcdf-4.3.2
./configure --prefix=$PYENV --enable-dap
make check
make install
cd ..

### install python packages
pip install numpy==1.9.3

### install netCDF4 python package
export HDF5_DIR=$PYENV
export NETCDF4_DIR=$PYENV
pip install netCDF4==1.2.4

### python package cffi and required libraries
sudo apt-get install python3-cffi
sudo apt-get install libffi-dev
pip install cffi==1.1.2

### required python packages
pip install h5py
pip install pandas==0.16.1
pip install matplotlib
pip install ipython[notebook]

# method to enable notebooks in WSL from http://blog.lanzani.nl/2016/jupyter-wsl/
pip uninstall pyzmq
sudo add-apt-repository ppa:aseering/wsl
sudo apt-get update
sudo apt-get install libzmq3 libzmq3-dev
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"
pip install --no-use-wheel -v pyzmq
pip install jupyter

# to start notebook server type at the bash command prompt: jupyter notebook
# then open a browser and point to http://localhost:8888/notebooks
```

Recommend installing AWRA CMS code bundle to bash file system to enable file permissions changes with chmod

Nosetests will only run python test files with all execution permissions turned off *chmod -x test_file.py*


#### Mac OS X

Recent versions of OS X comes pre-installed with Python 2.7, so you will need to separately install a more recent version (Python 3.5 or later). You will also need to install the HDF5 and NetCDF4 libraries and possibly the GDAL libraries. There are various options for installing each of these packages, but one of the easier options is to use the Homebrew package manager ([http://brew.sh](http://brew.sh)).

* Install Homebrew from [http://brew.sh](http://brew.sh).
* Install Python 3 using `brew install python3`
* Install NetCDF (which will install HDF5 as a dependency):
```
brew tap homebrew/science
brew install netcdf
```

* Optionally install GDAL using `brew install homebrew/versions/gdal111`
* Install the Python NetCDF4 library, the h5py library, pandas and the ipython notebook
```
pip3 install netcdf4 h5py pandas ipython[notebook]
```
* Optionally install GDAL Python package with `pip3 install gdal==1.11.1` (The version number is required because, at the time of writing, homebrew installs a version 1.X release of GDAL. If this changes and Homebrew installs a version 2.X library, you should be able to install the latest python bindings with `pip3 install gdal`).


### Installing the modelling systems from source

Download source from Github by either [cloning](https://github.com/awracms/awra_cms.git) or [downloading](https://github.com/awracms/awra_cms/archive/master.zip).
Then:
```
### download and install AWRA CMS
### activate virtual environment
. $PYENV/virtualenv/bin/activate

### if a zip file was downloaded
# wget https://github.com/awracms/awra_cms/archive/master.zip
unzip master.zip
cd awra_cms-master
### if a tarball was downloaded
# wget https://github.com/awracms/awra_cms/archive/master.tar.gz
tar zxf master.tar.gz
cd awra_cms-master
### if the repository was cloned
# git clone https://github.com/awracms/awra_cms.git
cd awra_cms

cd utils
python setup.py install
cd ..

cd benchmarking
python setup.py install
python setup.py nosetests
cd ..

cd models
python setup.py install
python setup.py nosetests
cd ..

cd simulation
python setup.py install
python setup.py nosetests
cd ..

cd visualisation
python setup.py install
python setup.py nosetests
cd ..

cd calibration
python setup.py install
python setup.py nosetests
cd ..

cd utils
python setup.py nosetests
cd ..

```
