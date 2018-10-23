import os
import shutil

os.system('clear')
#path = '/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/dataset/'

def dir_restore(path = './'):
    
    if os.access(path, os.F_OK) == True:
        
        # Controllo se la cartella 1_original_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + '1_original_dataset/', os.F_OK) == True:
            shutil.rmtree(path + '1_original_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + '1_original_dataset/')

            print('Cartella 1_original_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + '1_original_dataset/')
            print('Cartella 1_original_dataset creata')
        
        # Controllo se la cartella 2_reshaped_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + '2_reshaped_dataset/', os.F_OK) == True:
            shutil.rmtree(path + '2_reshaped_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + '2_reshaped_dataset/')

            print('Cartella 2_reshaped_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + '2_reshaped_dataset/')
            print('Cartella 2_reshaped_dataset creata')

        # Controllo se la cartella 3_image_dataset è prestente
        #  > Se è presente la elimino e la ricreo vuota
        #  > Se non è preswente la creo
        if os.access(path + '3_image_dataset/', os.F_OK) == True:
            shutil.rmtree(path + '3_image_dataset/', ignore_errors=False, onerror=None)
            
            os.mkdir(path + '3_image_dataset/')

            print('Cartella 3_image_dataset trovata, eliminata e ricreata')
        else:
            os.mkdir(path + '3_image_dataset/')
            print('Cartella 3_image_dataset creata')
        
    else:
        os.mkdir(path)
        os.mkdir(path + '1_original_dataset/')
        os.mkdir(path + '2_reshaped_dataset/')
        os.mkdir(path + '3_image_dataset/')


