import os
import json
from sklearn.metrics import confusion_matrix
import subprocess
import time
import gc

y_true = []
y_pred = []

dir = os.path.abspath("../../TestSet")
counts = {0:0,1:0,2:0,3:0}

for root, dirs, files in os.walk(dir):
    for filename in files:
        gc.collect()
        file = root + os.sep + filename
        # print(file)
        type_id_true = filename.split("_")[1]
        y_true.append(int(type_id_true))
        vuln_pred = 0
        pipe = os.popen("cd wap-2.1/wap-2.1/ && timeout 5s ./wap -a -sqli -s " + file )
        out = pipe.read()


        os.system("cd ..")
        os.system("cd ..")
        if "none" not in out and (vuln_pred == 0 or int(type_id_true) == 1):
            vuln_pred = 1

        pipe = os.popen("cd wap-2.1/wap-2.1/ && timeout 5s ./wap -a -xss -s " + file )
        out = pipe.read()

        os.system("cd ..")
        os.system("cd ..")
        if "none" not in out and (vuln_pred == 0 or int(type_id_true) == 2):
            vuln_pred = 2

        pipe = os.popen("cd wap-2.1/wap-2.1/ && timeout 5s ./wap -a -ci -s " + file)
        out = pipe.read()

        os.system("cd ..")
        os.system("cd ..")

        if "none" not in out and (vuln_pred == 0 or int(type_id_true) == 3):
            vuln_pred = 3

        y_pred.append(vuln_pred)



print(confusion_matrix(y_true,y_pred))
