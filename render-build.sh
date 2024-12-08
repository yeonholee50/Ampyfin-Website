#!/bin/bash

# Install build dependencies
apt-get update && apt-get install -y build-essential wget

# Download and extract TA-Lib source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib

# Build and install TA-Lib
./configure --prefix=/usr
make
sudo make install

# Return to the project root and install Python dependencies
cd ..
python -m pip install -r requirements.txt
