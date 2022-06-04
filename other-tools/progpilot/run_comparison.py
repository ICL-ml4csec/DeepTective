import os
import json
from sklearn.metrics import confusion_matrix

os.system("progpilot ../../TestSet/ >> results.json")

with open("results.json") as f:
    data = json.load(f)

#0 is safe, 1 is SQLi, 2 is XSS, 3 is CI
vuln_dict = {0:{0:0,1:0,2:0,3:0},1:{0:0,1:0,2:0,3:0},2:{0:0,1:0,2:0,3:0},3:{0:0,1:0,2:0,3:0}}

y_files=[]
y_true = []
y_pred = []

seen_files = set()

vuln_map = {"CWE_89":1,"CWE_79":2,"CWE_78":3}
vulns_in_scope = ["CWE_89","CWE_79","CWE_78"]

dir = "../../TestSet"

counts = {0:0,1:0,2:0,3:0}

for root, dirs, files in os.walk(dir):
    for filename in files:
        type_id_true = filename.split("_")[1]
        counts[int(type_id_true)] += 1

for detection in data:
    filename = os.path.basename(detection["source_file"][0])
    type_id_true = filename.split("_")[1]
    cwe = detection["vuln_cwe"]
    if cwe not in vulns_in_scope:
        continue
    type_id_pred = vuln_map[cwe]
    if filename in seen_files:
        if int(type_id_true) == int(type_id_pred):
            index_in = y_files.index(filename)
            y_pred[index_in] = int(type_id_pred)
            continue
        continue
    y_files.append(filename)
    y_true.append(int(type_id_true))
    y_pred.append(int(type_id_pred))
    seen_files.add(filename)
    counts[int(type_id_true)] -= 1

for i in range(4):
    y_true.extend([i] * counts[i])
    y_pred.extend([0] * counts[i])

print(confusion_matrix(y_true,y_pred))

os.system("rm results.json")