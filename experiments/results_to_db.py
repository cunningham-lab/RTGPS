import pandas as pd
from experiments.utils import load_results

input_file = './results/rff_ard_results/rffard_results.pkl'
results = load_results(input_file)
translator = {'pol': 'PolTele', 'ele': 'Elevators', 'kin': 'Kin40K', 'bike': 'Bike',
              'protein': 'Protein', 'keggu': 'KEGGU',
              '3droad': '3DRoad', 'keggd': 'KEGG'}

seeds = [5328, 49, 8965]
data = []
for model_name in results.keys():
    for model_type in results[model_name].keys():
        for dataset in results[model_name][model_type].keys():
            total = 3 if dataset == 'keggd' else 5
            for i in range(total):
                entries = {'Method': 'RFF',
                           'Dataset': translator[dataset]}
                for metric in ['seed', 'RMSE', 'NLL']:
                    val = results[model_name][model_type][dataset][metric][i]
                    metric = metric if not metric == 'seed' else 'Seed'
                    entries[metric] = val
                data.append(entries)
df = pd.DataFrame(data)
df = df[df['Seed'].isin(seeds)]
db = pd.read_csv('./paper_table/results.csv')
# db = db.drop('Unnamed: 0', axis=1)
# drop_idx = db.Method != 'RFF ARD'
drop_idx = db.Method != 'RFF'
db = db[drop_idx]
aux = db.append(df, ignore_index=True)
aux.to_csv('./paper_table/results.csv', index=True)
