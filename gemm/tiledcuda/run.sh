#!/bin/bash

if [ -d tmp ]; then
    rm -rf tmp
fi

python3 test.py 2>&1 | tee test.log
