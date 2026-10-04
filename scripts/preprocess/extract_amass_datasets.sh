#!/bin/bash

# Define target directory
SAVE_DIR="$HOME/data/amass"
DIR_OF_TAR_FILES="$HOME/Documents/amass_subsets"

cd $DIR_OF_TAR_FILES

for file in *.tar.bz2; do

  tar -xf "$file"  
  extracted_folder_name=$(basename "${file:0:-11}.tar.bz2" .tar.bz2)
  echo "Extracted folder name: $extracted_folder_name"
  echo "${SAVE_DIR}/${extracted_folder_name}"
  
  mv "$extracted_folder_name" "${SAVE_DIR}/${extracted_folder_name}"
done
