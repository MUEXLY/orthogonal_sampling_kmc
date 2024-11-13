#!/bin/bash

# This script takes a formatted kmc.pbs and creates multiple pbs files with different parameters
# It then prompts the user to inspect a randomly chosen file and to submit all files in a random order
# The script is interactive and will prompt the user for input
# It will get the template file from the current directory and save the generated files in a folder named pbs_output

input_file="kmc.pbs"
output_folder="pbs_output"
job_counter=0

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

mkdir -p ~/kmc_out # This is where error logs and stdout logs will be saved

# Clear the contents of the output folder
rm -f "$output_folder"/*

# Prompt the user for the number of sets of runs
read -p "Enter the number of sets of runs: " num_sets

# Loop through each set of runs
for run_set in $(seq 1 $num_sets); do
    # Loop through X values (1 to 5)
    for X in {1..5}; do
        # Loop through Z values corresponding to X
        case $X in
            1) Z="100";;
            2) Z="18,18";;
            3) Z="16,16,16";;
            4) Z="8,8,8,8";;
            5) Z="5,5,5,5,5";;
            *) Z="";;  # Default case, you can modify this based on your needs
        esac

        # Loop through Y values (0 to 0.14 with step 0.02)
        for Y in $(seq 0 0.02 0.14); do
            # Generate output file name
            output_file="$output_folder/output_X${X}_Y${Y//./_}_Z${Z//,/}_Set${run_set}.pbs"

            # Replace placeholders in the input file and save to the output file
            sed -e "s|X|$X|g; s|Y|$Y|g; s|Z|$Z|g; s|A|$run_set|g" "$input_file" > "$output_file"

            echo "Generated: $output_file"
            ((job_counter++))
        done
    done
done

# Display the number of jobs that will be created
echo "Total jobs to be created: $job_counter"

# Choose one file to inspect
chosen_file=$(ls "$output_folder"/*.pbs | shuf -n 1)

# Prompt the user if they want to inspect the chosen file
read -r -p "Do you want to inspect the file '$chosen_file' before submitting jobs? (y/n): " inspect_choice

if [ "$inspect_choice" == "y" ]; then
    # Use vim to inspect the file in read-only mode
    vim -R "$chosen_file"
fi

# Prompt the user for submission
read -r -p "Do you want to submit the generated files now? (y/n): " submit_choice

if [ "$submit_choice" == "y" ]; then
    # Submit all files in the output folder in a random order
    for file in $(shuf -e "$output_folder"/*.pbs); do
        qsub "$file"
    done
    echo "Files submitted in random order."
else
    echo "Files not submitted."
fi
