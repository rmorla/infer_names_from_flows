# infer_names_from_flows
Code to infer the domain name of an HTTPS flow given its tcp flow features using tensorflow. Learns from the topn most frequent domain names in the flow data. 


## Get flow data using tstat.polito.it
Capture some packets with tcpdump and process them with tstat, or capture with tstat directly. Use log_tcp_complete tstat data; convert to csv. http://tstat.polito.it/measure.shtml

## Install tensorflow
https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow:1.13.1-py3-jupyter
RUN pip install --upgrade pandas


## Build a dataset from tstat csv output
python infer_names_from_flows.py --do_what=build_dataset --tstat_csv=mytstatcapture.csv --tstat_npy=mytstatcapture.npy --topn=50

## Learn relation between domain name and flow stats
python infer_names_from_flows.py --do_what=learn --tstat_npy=mytstatcapture.npy --topn=50 --epochs=10
