import os
import json

root_dir = 'save/23.6.13/'
count = 0
result_list = []
with open('save/resultcsv/23.6.13.csv', 'w') as f:
    for root, dir_list, file_list in os.walk(root_dir):
        for file in file_list:
            result = json.loads(open(os.path.join(root, file)).read())
            result_list.append({
                'model': result['model'],
                'dataset': result['dataset'],
                'output_len': result['output_len'],
                'mse': result['mse'],
                'mae': result['mae'],
            })
            print(result['model'],
                  result['dataset'],
                  result['output_len'],
                  result['mse'],
                  result['mae'],
                  file=f,
                  sep=',')
