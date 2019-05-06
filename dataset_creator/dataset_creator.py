import os
import random
import foldertreemng
from tqdm import tqdm
from pathlib import Path
import matrix_to_image_fn

create_tree = foldertreemng.create_tree
clean_tree = foldertreemng.clean_tree

converter_fn = matrix_to_image_fn.converter_fn

def ms(bp, s, l, selestr=0.005, path='', i=24):
    
    comand = 'cd ' + path + ' ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(s) + ' -l ' + l + ' -selstr ' + str(selestr) + ' -p ' + path + '/ -i ' + str(i)
    print('comand: ' + comand)
    os.system(comand)
    
os.system('clear')


path = os.getcwd() + '/'
print('Path corrente: ' + path)

print('\nComando generico:\npython3 ms2raster.py -bp 10000 -s 2 -l selection -selstr 0.005 -p ' + path + ' -i 24\n')
print('-bp:\t\tlunghezza matrice')
print('-s:\t\tnumero dimatrici da generare')
print('-l:\t\tmodalità')
print("-selestr:\tntensita' della selezione si puo' controllare con -selstr (quando -l e' neutral, non viene considerata)")
print('-p:\t\tpath della cartella contenente i moduli necessari')
print('-i:\t\tnumero di individui presi in considerazione')

mod = input('\nAvviare in modalità SELECTION[S], NEUTRAL[N] o entrambi[B]? ')
bp = input('Inserire grandezza singole matrici (-bp): ')
selstr = input('Inserire un valore per selstr (-selstr): ')
n_train = int(input('Inserire quante matrici per il training set (-s): '))
n_eval = int(input("Inserire quante matrici per l'evaluatioin set (-s): "))
n_test = int(input('Inserire quante matrici per il test set (-s): '))
img = input('Vuoi trasformare le matrici in immagini? [Y]/[N]: ')
txt = input('Vuoi tenere le matrici in formato testuale? [Y]/[N]: ')
compr = input('\nVuoi comprimere il dataset appena creato? [Y]/[N]: ')
nome_archivio = 'DATASET'
if compr == 'Y':
    print(path)
    nome_archivio = input('Inserisci un nome da dare al file compresso: ')
    

if mod == 'B':
    n_train = n_train/2
    n_test = n_test/2
    n_eval = n_eval/2

create_tree(
    path=path, 
    mode=mod
    )

os.system('clear')

log = open(path + 'DATASET/log.txt', 'w')
log = open(path + 'DATASET/log.txt', 'a')
log.write('Caratteristiche del dataset:\n\n')
if mod == 'S':
    log.write('- Il dataset è stato eseguito creando solamente le matrici in modalità SELECTION.\n')
elif mod == 'N':
    log.write('- Il dataset è stato eseguito creando solamente le matrici in modalità NEUTRAL.\n')
else:
    log.write('- Il dataset è stato eseguito creando le matrici sia in modalità SELECTION che NEUTRAL.\n')

if mod == 'B':
    log.write('- ' + str(int(n_train*2)) + ' matrici per il training (' + str(int(n_train)) + ' per tipo)\n')
    log.write('- ' + str(int(n_test*2)) + ' matrici per il test (' + str(int(n_test)) + ' per tipo)\n')
    log.write("- " + str(int(n_eval*2)) + " matrici per l'evaluation (" + str(int(n_eval)) + " per tipo)\n")
else:
    log.write('- ' + str(int(n_train)) + ' matrici per il training')
    log.write('- ' + str(int(n_test)) + ' matrici per il test')
    log.write("- " + str(int(n_eval)) + " matrici per l'evaluation")
    
log.write('- le singole matrici hanno un bp di ' + bp + '\n')
log.write("- l'intensità della selezione è " + selstr + '\n')
log.write('- il numero di individui presi in considerazione è 24\n')
    
if mod == 'S' or mod == 'B':
    comand_train = 'cd ' + path + 'DATASET/SELECTION/TRAIN ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_train)) + ' -l selection -selstr ' + selstr + ' -p ' + path + 'DATASET/SELECTION/TRAIN/ -i +24'
    log.write('\n\nComando usato per generare il selection train set:\n' + comand_train + '\n\n')
    # print('[LOG:comand_train]: ' + comand_train)
    os.system(comand_train)   
    
    comand_test = 'cd ' + path + 'DATASET/SELECTION/TEST ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_test)) + ' -l selection -selstr ' + selstr + ' -p ' + path + 'DATASET/SELECTION/TEST/ -i +24'
    log.write('Comando usato per generare il selection test set:\n' + comand_test + '\n\n')
    # print('\nLOG:comand_test]: ' + comand_test)
    os.system(comand_test)

    comand_eval = 'cd ' + path + 'DATASET/SELECTION/EVAL ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_eval)) + ' -l selection -selstr ' + selstr + ' -p ' + path + 'DATASET/SELECTION/EVAL/ -i +24'
    log.write('Comando usato per generare il selection evaluation set:\n' + comand_eval + '\n\n')
    # print('\nLOG:comand_eval]: ' + comand_eval)
    os.system(comand_eval)   

if mod == 'N' or mod == 'B':
    comand_train = 'cd ' + path + 'DATASET/NEUTRAL/TRAIN ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_train)) + ' -l neutral -selstr ' + selstr + ' -p ' + path + 'DATASET/NEUTRAL/TRAIN/ -i +24'
    log.write('Comando usato per generare il neutral train set:\n' + comand_train + '\n\n')
    # print('[LOG:comand_train]: ' + comand_train)
    os.system(comand_train)   
    
    comand_test = 'cd ' + path + 'DATASET/NEUTRAL/TEST ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_test)) + ' -l neutral -selstr ' + selstr + ' -p ' + path + 'DATASET/NEUTRAL/TEST/ -i +24'
    log.write('Comando usato per generare il neutral test set:\n' + comand_test + '\n\n')
    # print('\nLOG:comand_test]: ' + comand_test)
    os.system(comand_test)

    comand_eval = 'cd ' + path + 'DATASET/NEUTRAL/EVAL ; python3 ms2raster.py -bp ' + str(bp) + ' -s ' + str(int(n_eval)) + ' -l neutral -selstr ' + selstr + ' -p ' + path + 'DATASET/NEUTRAL/EVAL/ -i +24'
    log.write('Comando usato per generare il neutral evaluation set:\n' + comand_eval + '\n\n')
    # print('\nLOG:comand_eval]: ' + comand_eval)
    os.system(comand_eval)

if img == 'Y': 

    if mod == 'S' or mod == 'B':
        os.mkdir(path + 'DATASET/SELECTION/TRAIN_IMG')
        os.mkdir(path + 'DATASET/SELECTION/TEST_IMG')
        os.mkdir(path + 'DATASET/SELECTION/EVAL_IMG')

        print('\nTrasformo il selection training set in immagini...')
        for i in tqdm(range(1, int(n_train)+1)):
            # SELECTION - TRAIN
            path_tmp_strain = path + 'DATASET/SELECTION/TRAIN/' + str(i) + '.selection.sim'
            file_strain = Path(path_tmp_strain)
            if file_strain.is_file():
                converter_fn(
                            path = path + 'DATASET/SELECTION/TRAIN/' + str(i) + '.selection.sim',
                            file_name = path + 'DATASET/SELECTION/TRAIN_IMG/' + str(i) + '.selection.png',
                            )

        print('\nTrasformo il selection test set in immagini...')
        for i in tqdm(range(1, int(n_test)+1)):
            # SELECTION - TEST
            path_tmp_stest = path + 'DATASET/SELECTION/TEST/' + str(i) + '.selection.sim'
            file_stest = Path(path_tmp_stest)
            if file_stest.is_file():
                converter_fn(
                            path = path + 'DATASET/SELECTION/TEST/' + str(i) + '.selection.sim',
                            file_name = path + 'DATASET/SELECTION/TEST_IMG/' + str(i) + '.selection.png',
                            )
        
        print('\nTrasformo il selection evaluation set in immagini...')
        for i in tqdm(range(1, int(n_eval)+1)):
            # SELECTION - EVAL
            path_tmp_seval = path + 'DATASET/SELECTION/EVAL/' + str(i) + '.selection.sim'
            file_seval = Path(path_tmp_seval)
            if file_seval.is_file():
                converter_fn(
                            path = path + 'DATASET/SELECTION/EVAL/' + str(i) + '.selection.sim',
                            file_name = path + 'DATASET/SELECTION/EVAL_IMG/' + str(i) + '.selection.png',
                            )
    if mod == 'N' or mod == 'B':
        os.mkdir(path + 'DATASET/NEUTRAL/TRAIN_IMG')
        os.mkdir(path + 'DATASET/NEUTRAL/TEST_IMG')
        os.mkdir(path + 'DATASET/NEUTRAL/EVAL_IMG')

        print('\nTrasformo il neutral training set in immagini...')
        for i in tqdm(range(1, int(n_train)+1)):
            # NEUTRAL - TRAIN
            path_tmp_ntrain = path + 'DATASET/NEUTRAL/TRAIN/' + str(i) + '.neutral.sim'
            file_ntrain = Path(path_tmp_ntrain)
            if file_ntrain.is_file():
                converter_fn(
                            path = path + 'DATASET/NEUTRAL/TRAIN/' + str(i) + '.neutral.sim',
                            file_name = path + 'DATASET/NEUTRAL/TRAIN_IMG/' + str(i) + '.neutral.png',
                            )

        print('\nTrasformo il neutral test set in immagini...')
        for i in tqdm(range(1, int(n_test)+1)):
            # NEUTRAL - TEST
            path_tmp_ntest = path + 'DATASET/NEUTRAL/TEST/' + str(i) + '.neutral.sim'
            file_ntest = Path(path_tmp_ntest)
            if file_ntest.is_file():
                converter_fn(
                            path = path + 'DATASET/NEUTRAL/TEST/' + str(i) + '.neutral.sim',
                            file_name = path + 'DATASET/NEUTRAL/TEST_IMG/' + str(i) + '.neutral.png',
                            )
        
        print("\nTrasformo il neutral evaluation set in immagini...")
        for i in tqdm(range(1, int(n_eval)+1)):
            # NEUTRAL - EVAL    
            path_tmp_neval = path + 'DATASET/NEUTRAL/EVAL/' + str(i) + '.neutral.sim'
            file_neval = Path(path_tmp_neval)
            if file_neval.is_file():
                converter_fn(
                            path = path + 'DATASET/NEUTRAL/EVAL/' + str(i) + '.neutral.sim',
                            file_name = path + 'DATASET/NEUTRAL/EVAL_IMG/' + str(i) + '.neutral.png',
                            )

    if txt == 'N':
        os.system('rm -r ' + path + 'DATASET/SELECTION/TEST')
        os.system('rm -r ' + path + 'DATASET/SELECTION/TRAIN')
        os.system('rm -r ' + path + 'DATASET/SELECTION/EVAL')

        os.system('rm -r ' + path + 'DATASET/NEUTRAL/TEST')
        os.system('rm -r ' + path + 'DATASET/NEUTRAL/TRAIN')
        os.system('rm -r ' + path + 'DATASET/NEUTRAL/EVAL')
    else:
        clean_tree(path=path, mode=mod)
    
if compr == 'Y':
    os.system('tar zcvf ' + nome_archivio + '.tar.gz DATASET')

log.close()
