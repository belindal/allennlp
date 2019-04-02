import json
import glob
import matplotlib.pyplot as plt
import pdb

annotations = []
percents = []
f1s = []
percent_to_f1 = {}
results_files = glob.glob("*.json")
for filename in results_files:
    file_percent = int(filename[:len(filename)-5])
    percents.append(file_percent)
    with open(filename, "r") as f:
        metric = json.load(f)
        annotations.append(metric["best_epoch"])
        f1s.append(metric["best_validation_coref_f1"])
        percent_to_f1[file_percent] = metric["best_validation_coref_f1"]

print(percent_to_f1)
plt.scatter(percents, f1s)
for i, txt in enumerate(annotations):
    plt.annotate(txt, (percents[i], f1s[i]))
plt.xlabel("% labels")
plt.ylabel("F1 score")
plt.show()
