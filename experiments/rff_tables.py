import numpy as np
from experiments.utils import TestMetrics
# from experiments.truncation_fns import compute_mean_of_trun
tm = TestMetrics()
# tm.load_db('./results/rff_results/test.pkl')
tm.load_db('./results/rff_ard_results/rffard_results.pkl')
translator = {
    'type_1': 'features - 1000',
    # 'type_1': 'features - 100',
    # 'type_2': 'features - 200',
    # 'type_3': 'features - 500',
    # 'type_4': 'features - 1000',
}
# tm.load_db('./results/ssrff_results/test.pkl')
# print(compute_mean_of_trun(coeff=0.5, min_val=100, max_val=1000, step=10))
# translator = {
#     'type_1': '0.5 - 10',
#     'type_2': '1.5 - 100',
#     'type_1': 'w / onej',
#     'type_2': 'no / onej',
#     'type_3': 'w / des - 75',
#     'type_4': 'no / des - 75',
#     'type_5': 'w / des - 90',
#     'type_6': 'no / des - 90',
# }
# ###########################################################################################
print(85 * '+')
for model, output in tm.db.items():
    for type_case, datasets in output.items():
        if type_case in translator.keys():
            print(model.upper() + ' ' + translator[type_case])
            for dataset_name, metrics in datasets.items():
                name_formated = dataset_name.rjust(15, ' ')
                text = name_formated
                for metric, values in metrics.items():
                    if not metric == 'seed':
                        aux = tm.db[model][type_case][dataset_name][metric]
                        text += ' ' + metric
                        text += f': {np.mean(aux): 4.3f}  +/- {np.std(aux): 4.5f} | '
                print(text)
print(85 * '+')
# ###########################################################################################
