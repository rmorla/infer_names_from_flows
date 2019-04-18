# infer_names_from_flows
Code to infer the domain name of a flow given its tcp flow features using tensorflow. Learns from the topn most frequent domain names in the tstat.polito.it data. 


## Build a dataset from tstat csv output
python infer_names_from_flows.py --do_what=build_dataset --tstat_csv=mytstatcapture.csv --tstat_npy=mytstatcapture.npy --topn=50

## Learn relation between domain name and flow stats
python infer_names_from_flows.py --do_what=learn --tstat_npy=mytstatcapture.npy --topn=50 --epochs=10
