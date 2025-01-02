#!/bin/bash

# Define the output file
output_file="cpu_frequency_log.txt"

# Number of seconds to wait between each check
interval=1

# Duration of logging (in seconds)
duration=600  # Adjust this to the duration you need

# Start time
end_time=$((SECONDS + duration))

# Loop to log CPU frequency every interval seconds until the duration is reached
while [ $SECONDS -lt $end_time ]; do
    /usr/sbin/hwinfo | grep "cpu MHz" >> $output_file
    sleep $interval
done
