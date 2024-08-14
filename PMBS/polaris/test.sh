#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A SR_APPFL 
#PBS -k doe
#PBS -N lat_test_test
#PBS -o lat_test_test.out
#PBS -e lat_test_test.err

# Get the list of allocated nodes from the PBS_NODEFILE
NODES=($(sort -u $PBS_NODEFILE))
SERVER_NODE=${NODES[0]}
CLIENT_NODE=${NODES[1]}

# Print out the hostnames of the allocated nodes to verify
pbsdsh -n 0 -- hostname
pbsdsh -n 1 -- hostname

# Server Node: Start the iperf server on the server node using pbsdsh
pbsdsh -n 0 -- /home/ldai8/ccl/run_server.sh

# Client Node: Run the iperf client on the client node using pbsdsh
pbsdsh -n 1 -- /home/ldai8/ccl/run_client.sh


