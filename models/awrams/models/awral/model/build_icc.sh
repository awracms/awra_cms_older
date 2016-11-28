#! /bin/sh

python template.py
icc awral.c -std=c99 -static-intel --shared -fPIC -vec-report3 -O3 -o awral.so
