import os
import shutil

os.system('clear')
#path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/msdir/'

def dir_restore(path = './', mode='AUTO'):
    
    if mode == 'AUTO':
        print('Funziona avviata in modalità AUTO')
        path = os.getcwd()
        path = path + '/msdir/'
        print(path)

    
    if os.access(path + 'dataset/', os.F_OK) == True:
        
        # Controllo se la cartella 1_original_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + 'dataset/1_original_dataset/', os.F_OK) == True:
            shutil.rmtree(path + 'dataset/1_original_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + 'dataset/1_original_dataset/')

            print('Cartella 1_original_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + 'dataset/1_original_dataset/')
            print('Cartella 1_original_dataset creata')
        
        # Controllo se la cartella 2_reshaped_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + 'dataset/2_reshaped_dataset/', os.F_OK) == True:
            shutil.rmtree(path + 'dataset/2_reshaped_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + 'dataset/2_reshaped_dataset/')

            print('Cartella 2_reshaped_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + 'dataset/2_reshaped_dataset/')
            print('Cartella 2_reshaped_dataset creata')

        # Controllo se la cartella 3_image_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + 'dataset/3_image_dataset/', os.F_OK) == True:
            shutil.rmtree(path + '/dataset/3_image_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + 'dataset/3_image_dataset/')

            print('Cartella 3_image_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + 'dataset/3_image_dataset/')
            print('Cartella 3_image_dataset creata')
        
    else:
        os.mkdir(path + 'dataset/')
        os.mkdir(path + 'dataset/1_original_dataset/')
        os.mkdir(path + 'dataset/2_reshaped_dataset/')
        os.mkdir(path + 'dataset/3_image_dataset/')


dir_restore('path','AUTO')