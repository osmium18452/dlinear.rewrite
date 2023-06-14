import os
import json

root_dir = 'save/linears/23.6.13.2.not.best.model'
count = 0
result_list = []
with open('save/resultcsv/linears.worst.csv', 'w') as f:
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
