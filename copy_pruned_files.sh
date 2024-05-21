#!/bin/bash

# Define the source base directory
SOURCE_BASE_DIR="./results"
# Define the target base directory
TARGET_BASE_DIR="/remote-home/share/research/mechinterp/phi-2-dictionary/results"
# Define the learning rate
LEARNING_RATE="2.5e-05"
# Define the languages
LANGUAGES=("zh" "en") # Add or modify languages as needed

# Loop through layers and languages
for LAYER in {1..17}; do
    for LANG in "${LANGUAGES[@]}"; do
        # Define the source and target directories
        SOURCE_DIR="${SOURCE_BASE_DIR}/L${LAYER}-${LANG}-${LEARNING_RATE}-phi/checkpoints"
        TARGET_DIR="${TARGET_BASE_DIR}/L${LAYER}-${LANG}-${LEARNING_RATE}-/checkpoints"
        
        # Ensure the target directory exists
        mkdir -p "${TARGET_DIR}"
        
        # Define the source and target file paths
        SOURCE_FILE="${SOURCE_DIR}/final.pt"
        TARGET_FILE="${TARGET_DIR}/final.pt"
        
        # Copy the file if it exists
        if [ -f "${SOURCE_FILE}" ]; then
            cp "${SOURCE_FILE}" "${TARGET_FILE}"
            echo "Copied ${SOURCE_FILE} to ${TARGET_FILE}"
        else
            echo "File ${SOURCE_FILE} does not exist"
        fi
    done
done
