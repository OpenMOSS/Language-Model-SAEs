#!/bin/bash

# Define the source base directory
SOURCE_BASE_DIR="./results"

# Define the target base directory
TARGET_BASE_DIR="/remote-home/share/research/mechinterp/phi-2-dictionary/results"

# Define the list of folders to copy
FOLDERS=("L0-zh-2.5e-05-phi" "L12-en-2.5e-05-phi" "L14-zh-2.5e-05-phi" "L1-en-2.5e-05-phi" "L28-zh-2.5e-05-phi" "L30-zh-2.5e-05-phi" "L5-en-2.5e-05-phi" "L7-zh-2.5e-05-phi"
         "L10-en-2.5e-05-phi" "L12-zh-2.5e-05-phi" "L15-en-2.5e-05-phi" "L1-zh-2.5e-05-phi" "L29-en-2.5e-05-phi" "L3-en-2.5e-05-phi" "L5-zh-2.5e-05-phi" "L8-en-2.5e-05-phi"
         "L10-zh-2.5e-05-phi" "L13-en-2.5e-05-phi" "L15-zh-2.5e-05-phi" "L27-en-2.5e-05-phi" "L29-zh-2.5e-05-phi" "L3-zh-2.5e-05-phi" "L6-en-2.5e-05-phi" "L8-zh-2.5e-05-phi"
         "L11-en-2.5e-05-phi" "L13-zh-2.5e-05-phi" "L16-en-2.5e-05-phi" "L27-zh-2.5e-05-phi" "L2-en-2.5e-05-phi" "L4-en-2.5e-05-phi" "L6-zh-2.5e-05-phi" "L9-en-2.5e-05-phi")

# Loop through each folder and copy to the target directory
for FOLDER in "${FOLDERS[@]}"; do
    # Define the source and target directories
    SOURCE_DIR="${SOURCE_BASE_DIR}/${FOLDER}"
    TARGET_DIR="${TARGET_BASE_DIR}/${FOLDER%-phi}"

    # Create the target directory if it doesn't exist
    mkdir -p "${TARGET_DIR}"

    # Copy the folder
    cp -r "${SOURCE_DIR}"/* "${TARGET_DIR}/"

    echo "Copied ${FOLDER} to ${TARGET_DIR}"
done

