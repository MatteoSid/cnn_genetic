import os
from tqdm import tqdm
import matrix_to_image_fn
from pathlib import Path
converter_fn = matrix_to_image_fn.converter_fn

os.system('clear')
path = os.getcwd() + '/'
print('[LOG:path]: ' + path)

for i in tqdm(range(1, 11)):
    path_tmp = path + 'SELECTION/TEST/' + str(i) + '.selection.sim'
    file_ = Path(path_tmp)

    if file_.is_file():
        converter_fn(
                    path = path + 'SELECTION/TEST/' + str(i) + '.selection.sim',
                    file_name = path + 'SELECTION/TEST_IMG/' + str(i) + '.selection.png',
                    )

