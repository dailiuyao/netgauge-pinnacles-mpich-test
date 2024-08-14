#!/bin/bash
# Load necessary modules if required
# module load perftest

echo "This is the client node: $(hostname)"
sleep 20  # Increase sleep time to ensure the server is ready
# Read the server IP address from the shared file
SERVER_HSN0_IP=$(cat /home/ldai8/ccl/server_ip.txt)
printf '[debug] SERVER_HSN0_IP is %s\n' "$SERVER_HSN0_IP"

# Test connectivity to the server
ping -c 4 $SERVER_HSN0_IP

# Run the iperf client, connecting to the server's hsn0 IP address
iperf -c $SERVER_HSN0_IP -p 5201 -t 10 -i 1 > iperf_client_output.log

# Adjusted command to extract and record the latency from the output
LATENCY=$(grep -oP "\[\d+\]\s+\d+\.\d+-\d+\.\d+\ssec\s+\d+\.\d+\sMBytes\s+\d+\.\d+\sMbits/sec\s+\d+\sms" iperf_client_output.log | awk '{print $NF}')
echo "Latency for one iteration: $LATENCY ms" > latency_result.txt
