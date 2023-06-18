import os
import json

root_dir = 'save/23.6.17/not.ind'
count = 0
result_list = []
with open('save/23.6.17.not.ind.csv', 'w') as f:
    for root, dir_list, file_list in os.walk(root_dir):
        for file in file_list:
            result = json.loads(open(os.path.join(root, file),encoding='utf8').read())
            result_list.append({
                'lr': result['lr'],
                'output_len': result['output_len'],
                'mse': result['mse'],
                'mae': result['mae'],
            })
            print(result['lr'],
                  result['output_len'],
                  result['mse'],
                  result['mae'],
                  result['save_step'],
                  file=f,
                  sep=',')
