#!/bin/bash

for file in $(find marimo/ -maxdepth 1 -type f -name "*.py" -not -name ".python-version"); do
    echo "Converting $file to ipynb"
    base_name=$(basename "$file" .py)
    output_file="jupyter/$base_name.ipynb"
    marimo export ipynb $file -o "$output_file"
    echo "Converted $file to ipynb ($output_file)"
done
