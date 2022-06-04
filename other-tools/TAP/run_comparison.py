import os
from sklearn.metrics import confusion_matrix

# TAP will require you to install it's dependencies

os.system("cd TAP && php Tokenizer.php >/dev/null 2>&1")
os.system("cd TAP && python tap.py >/dev/null 2>&1")

dir = os.path.abspath("../../TestSet")

with open("TAP/predict_result_testset.txt") as f:
    data = f.readlines()

content = [x.strip() for x in data]

y_true = []
y_pred = []
#
i = 0
for root, dirs, files in os.walk(dir):
    for filename in files:
        type_id_true = filename.split("_")[1]
        y_true.append(int(type_id_true))
        y_pred_id = int(content[i])
        # change TAP ids to our ids
        if y_pred_id == 1:
            y_pred_id = 3
        elif y_pred_id == 2:
            y_pred_id = 2
        elif y_pred_id == 3:
            y_pred_id = 1
        else:
            y_pred_id = 0
        y_pred.append(y_pred_id)
        i += 1

#

print(confusion_matrix(y_true,y_pred))

os.system("rm TAP/predict_result_testset.txt")
os.system("rm TAP/testset_tokens.txt")