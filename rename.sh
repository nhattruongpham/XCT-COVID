#!/bin/bash
# Declares variable with value set to 1
i=1 
# Scans each text file in the working directory
for file in *g; 
# Iterate the command below until all files are scanned
do 
  # Renames each file with "File" followed by incrementing number ($i)
	mv -- "$file" "img_$i.png"
  # Increments the variables current number value by 1
  i=$((i+1)) 
done
