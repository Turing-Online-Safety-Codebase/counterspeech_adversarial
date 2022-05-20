#!/bin/sh
python3 -u script.py > logs/script.log
az account set --subscription b0c0051c-9dc9-4824-9c4d-5a0bb1bc8251
az vm deallocate -n cs-vm-1 -g cs-1
