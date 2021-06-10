from experiments.utils import TestMetrics
# datasets = {1: 'pol', 2: 'ele', 3: 'bike', 4: 'kin', 5: 'protein',
#             6: 'keggd', 7: 'slice', 8: 'keggu', 9: '3droad', 11: 'buzz'}
datasets = {1: 'pol', 2: 'ele', 3: 'bike', 4: 'kin', 5: 'protein',
            6: 'keggd', 8: 'keggu', 9: '3droad'}
tm = TestMetrics()

translator = {
    'type_1': {'rff_samples': 1000},
    # 'type_1': {'rff_samples': 100},
    # 'type_2': {'rff_samples': 200},
    # 'type_3': {'rff_samples': 500},
    # 'type_4': {'rff_samples': 1000},
}
consider_only = 0
tm.create_db(model_name='rff_ard', types=len(translator.keys()),
             metrics=['RMSE', 'NLL'], datasets=datasets)
# tm.populate_db(lookup_path='./results/rff_results/test/', key='rff',
#                translator=translator)
# output_file = './results/rff_results/test.pkl'
tm.populate_db(lookup_path='./results/rff_ard_results/test/', key='rff_ard',
               translator=translator)
output_file = './results/rff_ard_results/test.pkl'

# translator = {
#     'type_1': {'coeff': 0.5, 'step': 10},
#     'type_2': {'coeff': 1.5, 'step': 100},
# }
# lookup_path = './results/ssrff_results/test/'
# consider_only = 3
# total_types = len(translator)
# tm.create_db('ssrff', total_types, ['RMSE', 'NLL'], datasets)
# tm.populate_db(lookup_path=lookup_path, key='ssrff', translator=translator)
# output_file = './results/ssrff_results/test.pkl'

tm.convert_to_np(consider_only)
tm.save_db(output_file=output_file)
