#!/bin/bash

basedir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

cd $basedir/output  \
    && cmake ../    \
    && cmake --build . --config Release
