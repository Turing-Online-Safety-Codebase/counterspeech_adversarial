# Counter speech classification using adversarial training
 
## Collect abusive tweets and their replies

## Data preprosessing

### Transform the collected data into long format

### Data annotation and analysis


## Model training

To train counter speech classifiers, run

```
bash ./scripts/train_model.sh
```


## Update adversarial examples to training data


## Running and deallocating on Azure VM

The `run_python_then_dealloc.sh` shell script provides a simple way to have an Azure VM automatically deallocate after a python script finishes.

Edit the name of the python script to be run, the name of the log file to write to, and the name of the VM to deallocate, to match your use case. 

Before running it for the first time, you must run the following command:

`chmod +x run_python_then_dealloc.sh`

Then, to run the script in the background (so it will continue to run after exiting the VM), run the following:

`nohup ./run_python_then_dealloc.sh &`
