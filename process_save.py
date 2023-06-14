import os
import json

root_dir = 'save/formers/23.6.13.1.worst.model'
count = 0
result_list = []
with open('results/fromers.worst.csv', 'w') as f:
    for root, dir_list, file_list in os.walk(root_dir):
        for file in file_list:
            result = json.loads(open(os.path.join(root, file)).read())
            result_list.append({
                'model': result['model']+('.individual' if result['individual'] else ''),
                'dataset': result['dataset'],
                'output_len': result['output_len'],
                'mse': result['mse'],
                'mae': result['mae'],
            })
            print(result['model']+('.individual' if result['individual'] else ''),
                  result['dataset'],
                  result['output_len'],
                  result['mse'],
                  result['mae'],
                  file=f,
                  sep=',')
