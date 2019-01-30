import os
import random
import foldertreeemng
from tqdm import tqdm
from pathlib import Path
import matrix_to_image_fn

create_tree = foldertreeemng.create_tree
clean_tree = foldertreeemng.clean_tree

converter_fn = matrix_to_image_fn.converter_fn

def ms(bp, s, l, selestr=0.005, path='', i=24):
    
    comand = 'cd ' + path + ' ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(s) + ' -l ' + l + ' -selstr ' + str(selestr) + ' -p ' + path + '/ -i ' + str(i)
    print('comand: ' + comand)
    os.system(comand)

os.system('rm -rf *.sim')
os.system('clear')


path = os.getcwd() + '/'
print('Path corrente: ' + path)

print('\nComando generico:\npython3 ms2raster.py -bp 10000 -s 2 -l selection -selstr 0.005 -p ' + path + ' -i 24\n')
print('-bp:\t\tlunghezza matrice')
print('-s:\t\tnumero dimatrici da generare')
print('-l:\t\tmodalità')
print("-selestr:\tntensita' della selezione si puo' controllare con -selstr (quando -l e' neutral, non viene considerata)")
print('-p:\t\tpath della cartella contenente i moduli necessari')
print('-i:\t')

mod = input('\nAvviare in modalità SELECTION[S], NEUTRAL[N] o entrambi[B]? ')
n_mat = int(input('Inserire grandezza singole matrici (-bp): '))
n_train = int(input('Inserire quante matrici per il training set (-s): '))
n_test = int(input('Inserire quante matrici per il test set (-s): '))
selstr = input('Inserire un valore per selstr (-selstr): ')

create_tree(
    path=path, 
    mode=mod
    )

os.system('clear')
    
if mod == 'S' or mod == 'B':
    comand_train = 'cd ' + path + 'DATASET/SELECTION/TRAIN ; python3 ms2raster.py -bp ' + str(n_mat) + ' -s ' + str(n_train) + ' -l selection -selstr ' + selstr + ' -p ' + path + 'DATASET/SELECTION/TRAIN/ -i +24'
    print('[LOG:comand_train]: ' + comand_train)
    # os.system(comand_train)   
    
    comand_test = 'cd ' + path + 'DATASET/SELECTION/TEST ; python3 ms2raster.py -bp ' + str(n_mat) + ' -s ' + str(n_test) + ' -l selection -selstr ' + selstr + ' -p ' + path + 'DATASET/SELECTION/TEST/ -i +24'
    print('\nLOG:comand_test]: ' + comand_test)
    # os.system(comand_test)   

if mod == 'N' or mod == 'B':
    comand_train = 'cd ' + path + 'DATASET/NEUTRAL/TRAIN ; python3 ms2raster.py -bp ' + str(n_mat) + ' -s ' + str(n_train) + ' -l neutral -selstr ' + selstr + ' -p ' + path + 'DATASET/NEUTRAL/TRAIN/ -i +24'
    print('[LOG:comand_train]: ' + comand_train)
    # os.system(comand_train)   
    
    comand_test = 'cd ' + path + 'DATASET/NEUTRAL/TEST ; python3 ms2raster.py -bp ' + str(n_mat) + ' -s ' + str(n_test) + ' -l neutral -selstr ' + selstr + ' -p ' + path + 'DATASET/NEUTRAL/TEST/ -i +24'
    print('\nLOG:comand_test]: ' + comand_test)
    # os.system(comand_test)

if input('\nVuoi trasformare le matrici in immagini? [Y]/[N]: ') == 'Y': 

    if mod == 'S' or mod == 'B':
        os.mkdir(path + 'DATASET/SELECTION/TRAIN_IMG')
        os.mkdir(path + 'DATASET/SELECTION/TEST_IMG')

        print('\nTrasformo il selection training set in immagini...')
        for i in tqdm(range(1, n_train+1)):
            # SELECTION - TRAIN
            path_tmp_strain = path + 'DATASET/SELECTION/TRAIN/' + str(i) + '.selection.sim'
            file_strain = Path(path_tmp_strain)
            if file_strain.is_file():
                converter_fn(
                            path = path + 'DATASET/SELECTION/TRAIN/' + str(i) + '.selection.sim',
                            file_name = path + 'DATASET/SELECTION/TRAIN_IMG/' + str(i) + '.selection.png',
                            )

        print('\nTrasformo il selection test set in immagini...')
        for i in tqdm(range(1, n_test+1)):
            # SELECTION - TEST
            path_tmp_stest = path + 'DATASET/SELECTION/TEST/' + str(i) + '.selection.sim'
            file_stest = Path(path_tmp_stest)
            if file_stest.is_file():
                converter_fn(
                            path = path + 'DATASET/SELECTION/TEST/' + str(i) + '.selection.sim',
                            file_name = path + 'DATASET/SELECTION/TEST_IMG/' + str(i) + '.selection.png',
                            )
    if mod == 'N' or mod == 'B':
        os.mkdir(path + 'DATASET/NEUTRAL/TRAIN_IMG')
        os.mkdir(path + 'DATASET/NEUTRAL/TEST_IMG')

        print('\nTrasformo il neutral training set in immagini...')
        for i in tqdm(range(1, n_train+1)):
            # NEUTRAL - TRAIN
            path_tmp_ntrain = path + 'DATASET/NEUTRAL/TRAIN/' + str(i) + '.neutral.sim'
            file_ntrain = Path(path_tmp_ntrain)
            if file_ntrain.is_file():
                converter_fn(
                            path = path + 'DATASET/NEUTRAL/TRAIN/' + str(i) + '.neutral.sim',
                            file_name = path + 'DATASET/NEUTRAL/TRAIN_IMG/' + str(i) + '.neutral.png',
                            )

        print('\nTrasformo il neutral test set in immagini...')
        for i in tqdm(range(1, n_test+1)):
            # NEUTRAL - TEST
            path_tmp_ntest = path + 'DATASET/NEUTRAL/TEST/' + str(i) + '.neutral.sim'
            file_ntest = Path(path_tmp_ntest)
            if file_ntest.is_file():
                converter_fn(
                            path = path + 'DATASET/NEUTRAL/TEST/' + str(i) + '.neutral.sim',
                            file_name = path + 'DATASET/NEUTRAL/TEST_IMG/' + str(i) + '.neutral.png',
                            )

    if input('\nVuoi tenere le matrici in formato testuale? [Y]/[N]: ') == 'N':
        os.system('rm -r ' + path + 'DATASET/SELECTION/TEST')
        os.system('rm -r ' + path + 'DATASET/SELECTION/TRAIN')

        os.system('rm -r ' + path + 'DATASET/NEUTRAL/TEST')
        os.system('rm -r ' + path + 'DATASET/NEUTRAL/TRAIN')
else:
    clean_tree(path=path, mode=mod)

if input('\nVuoi comprimere il dataset appena creato? [Y]/[N]: ') == 'Y':
    print(path)
    # nome_archivio = input('Inserisci un nome da dare al file compresso: ')
    # os.system('tar zcvf ' + nome_archivio + '.tar.gz ' + path)
