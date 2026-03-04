

import json
import os

kaggle_dict = {
    "username": "Sathya Mozhi",
    "key": "KGAT_a441b02dfc006e7ccb28cb7260013171"
}

with open("kaggle.json", "w") as file:
    json.dump(kaggle_dict, file)

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.replace("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

print("Kaggle API configured successfully!")