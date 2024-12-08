#!/bin/bash

# Install build dependencies
tar zxvf ta-lib-0.4.0-src.tar.gz

# Download and extract TA-Lib source
cd ta-lib

# Build and install TA-Lib
./configure --prefix=/usr
make
sudo make install

# Return to the project root and install Python dependencies
cd ..
python -m pip install -r requirements.txt
