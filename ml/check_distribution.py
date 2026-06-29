import numpy as np
import json
import os

ML_DIR = os.path.dirname(os.path.abspath(__file__))

y_train = np.load(os.path.join(ML_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(ML_DIR, "y_test.npy"))
y_all   = np.concatenate([y_train, y_test])

labels = {0: "Away Win", 1: "Home Win", 2: "Draw"}
total  = len(y_all)

print(f"Total meciuri: {total}")
print()
for cls, name in labels.items():
    count = int(np.sum(y_all == cls))
    pct   = count / total * 100
    print(f"  {name} ({cls}): {count} ({pct:.1f}%)")
