#! /bin/sh

python template.py
gcc awral.c --std=c99 --shared -fPIC -O4 -o awral.so
