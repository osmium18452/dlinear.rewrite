import os
import json

root_dir='save/23.6.13/'
count=0
for root, dir_list,file_list in os.walk(root_dir):
    for file in file_list:
        result=json.loads(open(os.path.join(root,file)).read())
        print(result['mae'])
        count+=1
print(count)
