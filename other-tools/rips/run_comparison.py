import os
import json
from sklearn.metrics import confusion_matrix

dir = os.path.abspath("../../TestSet")

os.system("php rips-0.55/rips-0.55/main.php ignore_warning=true vector=all loc=" + dir + ">> /dev/null")

with open("results.json") as f:
    data = json.load(f)

#0 is safe, 1 is SQLi, 2 is XSS, 3 is CI
vuln_dict = {0:{0:0,1:0,2:0,3:0},1:{0:0,1:0,2:0,3:0},2:{0:0,1:0,2:0,3:0},3:{0:0,1:0,2:0,3:0}}

y_true = []
y_pred = []

seen_files = set()

vuln_map = {"SQL Injection":1,"Cross-Site Scripting":2,"Command Execution":3}
vulns_in_scope = ["SQL Injection","Cross-Site Scripting","Command Execution"]



counts = {0:0,1:0,2:0,3:0}

for root, dirs, files in os.walk(dir):
    for filename in files:
        file = root + os.sep + filename
        type_id_true = filename.split("_")[1]
        if file not in data:
            y_true.append(int(type_id_true))
            y_pred.append(0)
            continue
        detections = data[file]
        vuln_pred = 0
        for detection in detections:
            cat = detection["category"]
            # check if any of the predictions are actual, otherwise take first prediction in scope
            if cat in vulns_in_scope and (vuln_pred == 0 or int(vuln_map[cat]) == int(type_id_true)):
                vuln_pred = vuln_map[cat]
        y_true.append(int(type_id_true))
        y_pred.append(int(vuln_pred))


print(confusion_matrix(y_true,y_pred))

os.system("rm results.json")