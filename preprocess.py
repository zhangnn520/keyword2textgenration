import glob
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd

# ------------------------------------------------- #
#            Preprocessing the data                 #
# ------------------------------------------------- #
files = glob.glob(os.getcwd() + "/data/en/train/**/*.xml", recursive=True)
print(files)

triple_re = re.compile('(\d)triples')
data_dct = {}

for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num = int(triple_re.findall(file)[0])
    for sub_root in root:
        for ss_root in sub_root:
            strutured_master = []
            unstructured = []
            for entry in ss_root:
                unstructured.append(entry.text)
                strutured = [triple.text for triple in entry]
                strutured_master.extend(strutured)
            unstructured = [i for i in unstructured if i.replace('\n', '').strip() != '']
            strutured_master = strutured_master[-triples_num:]
            strutured_master_str = (' && ').join(strutured_master)
            data_dct[strutured_master_str] = unstructured
mdata_dct = {"prefix": [], "input_text": [], "target_text": []}
for st, unst in data_dct.items():
    for i in unst:
        mdata_dct['prefix'].append('webNLG')
        mdata_dct['input_text'].append(st)
        mdata_dct['target_text'].append(i)

df = pd.DataFrame(mdata_dct)
path_data = os.path.join("data", 'webNLG2020_train.csv')
df.to_csv(path_data)
