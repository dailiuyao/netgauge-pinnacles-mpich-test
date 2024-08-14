#!/bin/bash
# Load necessary modules if required
# module load perftest

echo "This is the server node: $(hostname)"
# Get the IP address associated with the hsn0 NIC
HSN0_IP=$(ip addr show hsn0 | grep "inet " | awk '{print $2}' | cut -d/ -f1)
printf '[debug] HSN0_IP on SERVER_NODE is %s\n' "$HSN0_IP"
# Save the IP address to a file accessible to both nodes
echo $HSN0_IP > /home/ldai8/ccl/server_ip.txt
# Start the iperf server bound to the hsn0 IP address
iperf -s -B $HSN0_IP -p 5201 &
sleep 10  # Allow some time for the server to start
# Verify if the iperf server is running
if pgrep -f "iperf -s"; then
    echo "iperf server started successfully"
else
    echo "iperf server failed to start"
fi
