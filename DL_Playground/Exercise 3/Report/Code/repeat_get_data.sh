#!/bin/bash
for i in {1..1000}
do
    echo "$i "
    ./get_data.py
done
