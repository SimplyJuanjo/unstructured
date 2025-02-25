#!/usr/bin/env bash

# Clean the content of json file generated by unstructured library, storing just 
# text elements. The resulting file will be stored at the $2 folder with the same
# name as the original file appending .txt as suffix.
# Arguments:
# - $1 path to the file to clean
# - $2 path to folder to store the result
# 

BASE=$(basename "$1")
DEST=$2/$BASE.txt
jq '.[].text'<"$1"|fold -w 80 -s > "$DEST"
