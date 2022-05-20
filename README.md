# counterspeech_MPs

## Running and deallocating

The `run_python_then_dealloc.sh` shell script provides a simple way to have an Azure VM automatically after a python script finishes.

Edit the name of the python script to be run, the name of the log file to write to, and the name of the VM to deallocate, to match your use case. 

Before running it for the first time, you must run the following command:

`chmod +x run_python_then_dealloc.sh`

Then, to run the script in the background (so it will continue to run after exiting the VM), run the following:

`nohup ./run_python_then_dealloc.sh &`
