import pickle
import sys
import json
import pandas as pd

#print(sys.argv)
if len(sys.argv) != 3:
    raise Exception("usage: python pkl2json.py <score list file> <out csv file>")

pkl_file = sys.argv[1]
out_file = sys.argv[2]

pkl = pickle.load(open(pkl_file, "rb"))

print("writing csv to {} ...".format(out_file))
with open(out_file, "w") as f:
    f.write(pd.DataFrame(pkl).to_csv())