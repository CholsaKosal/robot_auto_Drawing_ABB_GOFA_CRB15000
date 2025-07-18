#!/bin/bash

# ==============================================================================
# WSL-Optimized Code Base Generation Script (Config-Driven)
# ==============================================================================
# This script is designed to run within WSL, especially when the project
# directory is on a Windows-mounted filesystem (e.g., /mnt/c, /mnt/d).
#
# Key Improvements from previous version:
# - ADDED: Logic to automatically strip trailing slashes from patterns in the
#          config file, preventing common 'find' command warnings.
# - ADDED: The temporary file is now created before the loop, preventing a
#          "No such file or directory" error if no files are found.
# ==============================================================================

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
shopt -s nullglob # Globs that match nothing expand to nothing

OUTPUT_FILE="code_base_wsl.txt"
TEMP_CONTENT_FILE="wsl_temp_all_contents.txt"
EXCLUSIONS_CONFIG="exclude_patterns.conf"
WSL_EXCLUSIONS_CONFIG="exclude_wsl_patterns.conf"
SCRIPT_NAME="wsl_generate_code_base.sh"

# --- Helper Functions ---
print_info() {
  echo -e "\033[34m[INFO]\033[0m $1"
}

print_warning() {
  echo -e "\033[33m[WARNING]\033[0m $1"
}

print_error() {
  echo -e "\033[31m[ERROR]\033[0m $1" >&2
}

# --- Script Start ---
print_info "--- WSL-Optimized Script Start (Config-Driven) ---"
print_info "Output file: $OUTPUT_FILE"

# Clean up previous run's files
rm -f "$OUTPUT_FILE" "$TEMP_CONTENT_FILE"

# --- Build Exclusion List ---
# Determine which config file to use
if [ -f "$WSL_EXCLUSIONS_CONFIG" ]; then
  print_info "Using WSL-specific exclusion file: $WSL_EXCLUSIONS_CONFIG"
  FINAL_EXCLUSIONS_CONFIG="$WSL_EXCLUSIONS_CONFIG"
else
  print_info "WSL-specific config not found. Falling back to: $EXCLUSIONS_CONFIG"
  FINAL_EXCLUSIONS_CONFIG="$EXCLUSIONS_CONFIG"
fi

# Minimal patterns to prevent the script from processing its own files.
EXCLUDE_PATTERNS=(
  "./$OUTPUT_FILE"
  "./$TEMP_CONTENT_FILE"
  "./$SCRIPT_NAME"
  "./$EXCLUSIONS_CONFIG"
  "./$WSL_EXCLUSIONS_CONFIG"
)

# Check for and read the user-defined exclusion file
if [ ! -f "$FINAL_EXCLUSIONS_CONFIG" ]; then
  print_error "Exclusion config file '$FINAL_EXCLUSIONS_CONFIG' not found. Please create it and add patterns to exclude."
  exit 1
fi

print_info "Reading exclusion patterns from $FINAL_EXCLUSIONS_CONFIG..."
# Read file line by line to handle spaces correctly
while IFS= read -r line; do
  # Ignore empty lines and comments
  if [[ -n "$line" && ! "$line" =~ ^\s*# ]]; then
    # FIX: Remove trailing slashes from the pattern to prevent 'find' warnings
    line_no_slash="${line%/}"
    # Ensure patterns are treated as relative to the current directory
    EXCLUDE_PATTERNS+=("./$line_no_slash")
  fi
done < "$FINAL_EXCLUSIONS_CONFIG"
print_info "Finished reading $FINAL_EXCLUSIONS_CONFIG."


# --- Generate Project Tree ---
print_info "Generating project tree..."
print_warning "This may take a moment on a Windows filesystem..."

# Build the exclude pattern for the 'tree' command
# 'sed' removes the leading './' and the trailing '|' for the 'tree' command format
TREE_EXCLUDE_PATTERN=$(printf "%s|" "${EXCLUDE_PATTERNS[@]}" | sed 's|./||g' | sed 's/|$//')
tree -a -I "$TREE_EXCLUDE_PATTERN" > "$OUTPUT_FILE"
print_info "Tree generation complete."

# --- Gather File Contents ---
echo -e "\n\n=======================================\n        FILE CONTENTS START HERE         \n=======================================\n" >> "$OUTPUT_FILE"

print_info "Gathering file contents..."
print_warning "This is the slowest step. Please be patient. The script is working."

# FIX: Create the temp file before the loop to prevent "No such file" error
touch "$TEMP_CONTENT_FILE"

# Build the 'find' command arguments for exclusion
FIND_EXCLUDE_ARGS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  # -prune works on directories, so we use it for paths
  FIND_EXCLUDE_ARGS+=(-o -path "$pattern" -prune)
done

# The find command:
# - Starts at the current directory '.'
# - The first `-path` is a dummy to allow the first `-o`
# - It then iterates through our exclusion list, adding `-o -path '...' -prune' for each
# - Finally, it prints the files that were NOT pruned.
find . -path 'a/b/c/d' ${FIND_EXCLUDE_ARGS[@]} -o -type f -print0 | while IFS= read -r -d $'\0' file; do
    # Skip files that are likely binary or not useful text
    if [[ "$file" == *.png || "$file" == *.jpg || "$file" == *.jpeg || "$file" == *.gif || "$file" == *.webp || "$file" == *.svg || "$file" == *.ico || "$file" == *.map ]]; then
        continue
    fi
    
    # Check if the file is not empty
    if [ -s "$file" ]; then
        echo "--- START OF FILE: ${file#./} ---" >> "$TEMP_CONTENT_FILE"
        cat "$file" >> "$TEMP_CONTENT_FILE"
        echo -e "\n--- END OF FILE: ${file#./} ---\n" >> "$TEMP_CONTENT_FILE"
    fi
done

print_info "Finished gathering file contents."

# Append collected contents to the main output file
cat "$TEMP_CONTENT_FILE" >> "$OUTPUT_FILE"
print_info "Appended contents to $OUTPUT_FILE."

# --- Cleanup ---
rm "$TEMP_CONTENT_FILE"
print_info "Removed temporary file."
print_info "--- Script Finished Successfully ---"
