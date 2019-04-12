import os

def create_tree(path, mode):
    if os.access(path + 'DATASET', os.F_OK) == False:
                os.mkdir(path + 'DATASET')

    if mode == 'S' or mode == 'B':

        if os.access(path + 'DATASET/SELECTION', os.F_OK) == True:
            os.system('rm -r ' + path + 'DATASET/SELECTION')
            os.mkdir(path + 'DATASET/SELECTION')
        else:
            os.mkdir(path + 'DATASET/SELECTION')

        os.mkdir(path + 'DATASET/SELECTION/TRAIN/')
        os.mkdir(path + 'DATASET/SELECTION/TEST/')
        os.mkdir(path + 'DATASET/SELECTION/EVAL/')

        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/SELECTION/TRAIN')
        os.system('cd ' + path + 'DATASET/SELECTION/TRAIN' + ' ; unzip ms2raster.zip')
        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/SELECTION/TEST')
        os.system('cd ' + path + 'DATASET/SELECTION/TEST' + ' ; unzip ms2raster.zip')
        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/SELECTION/EVAL')
        os.system('cd ' + path + 'DATASET/SELECTION/EVAL' + ' ; unzip ms2raster.zip')

    if mode == 'N' or mode == 'B':

        if os.access(path + 'DATASET/NEUTRAL', os.F_OK) == True:
            os.system('rm -r ' + path + 'DATASET/NEUTRAL')
            os.mkdir(path + 'DATASET/NEUTRAL')
        else:
            os.mkdir(path + 'DATASET/NEUTRAL')

        os.mkdir(path + 'DATASET/NEUTRAL/TRAIN/')
        os.mkdir(path + 'DATASET/NEUTRAL/TEST/')
        os.mkdir(path + 'DATASET/NEUTRAL/EVAL/')
        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/NEUTRAL/TRAIN')
        os.system('cd ' + path + 'DATASET/NEUTRAL/TRAIN' + ' ; unzip ms2raster.zip')
        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/NEUTRAL/TEST')
        os.system('cd ' + path + 'DATASET/NEUTRAL/TEST' + ' ; unzip ms2raster.zip')
        os.system('cp ' + path + 'ms2raster.zip ' + path + 'DATASET/NEUTRAL/EVAL')
        os.system('cd ' + path + 'DATASET/NEUTRAL/EVAL' + ' ; unzip ms2raster.zip')

def clean_tree(path, mode):
    if mode == 'S' or mode =='B':

        path_tmp = path + 'DATASET/SELECTION/'

        os.system('cd ' + path_tmp + 'TRAIN ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'TRAIN ; rm ms')
        os.system('cd ' + path_tmp + 'TRAIN ; rm lastp0')
        os.system('cd ' + path_tmp + 'TRAIN ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'TRAIN ; rm mssel')
        os.system('cd ' + path_tmp + 'TRAIN ; rm seedms')
        os.system('cd ' + path_tmp + 'TRAIN ; rm stepftn')
        os.system('cd ' + path_tmp + 'TRAIN ; rm tp.out')
        os.system('cd ' + path_tmp + 'TRAIN ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'TRAIN ; rm -r __MACOSX')

        os.system('cd ' + path_tmp + 'TEST ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'TEST ; rm ms')
        os.system('cd ' + path_tmp + 'TEST ; rm lastp0')
        os.system('cd ' + path_tmp + 'TEST ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'TEST ; rm mssel')
        os.system('cd ' + path_tmp + 'TEST ; rm seedms')
        os.system('cd ' + path_tmp + 'TEST ; rm stepftn')
        os.system('cd ' + path_tmp + 'TEST ; rm tp.out')
        os.system('cd ' + path_tmp + 'TEST ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'TEST ; rm -r __MACOSX')

        os.system('cd ' + path_tmp + 'EVAL ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'EVAL ; rm ms')
        os.system('cd ' + path_tmp + 'EVAL ; rm lastp0')
        os.system('cd ' + path_tmp + 'EVAL ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'EVAL ; rm mssel')
        os.system('cd ' + path_tmp + 'EVAL ; rm seedms')
        os.system('cd ' + path_tmp + 'EVAL ; rm stepftn')
        os.system('cd ' + path_tmp + 'EVAL ; rm tp.out')
        os.system('cd ' + path_tmp + 'EVAL ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'EVAL ; rm -r __MACOSX')
    
    if mode == 'N' or mode == 'B':

        path_tmp = path + 'DATASET/NEUTRAL/'

        os.system('cd ' + path_tmp + 'TRAIN ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'TRAIN ; rm ms')
        os.system('cd ' + path_tmp + 'TRAIN ; rm lastp0')
        os.system('cd ' + path_tmp + 'TRAIN ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'TRAIN ; rm mssel')
        os.system('cd ' + path_tmp + 'TRAIN ; rm seedms')
        os.system('cd ' + path_tmp + 'TRAIN ; rm stepftn')
        os.system('cd ' + path_tmp + 'TRAIN ; rm tp.out')
        os.system('cd ' + path_tmp + 'TRAIN ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'TRAIN ; rm -r __MACOSX')

        os.system('cd ' + path_tmp + 'TEST ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'TEST ; rm ms')
        os.system('cd ' + path_tmp + 'TEST ; rm lastp0')
        os.system('cd ' + path_tmp + 'TEST ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'TEST ; rm mssel')
        os.system('cd ' + path_tmp + 'TEST ; rm seedms')
        os.system('cd ' + path_tmp + 'TEST ; rm stepftn')
        os.system('cd ' + path_tmp + 'TEST ; rm tp.out')
        os.system('cd ' + path_tmp + 'TEST ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'TEST ; rm -r __MACOSX')

        os.system('cd ' + path_tmp + 'EVAL ; rm ms2raster.zip')
        os.system('cd ' + path_tmp + 'EVAL ; rm ms')
        os.system('cd ' + path_tmp + 'EVAL ; rm lastp0')
        os.system('cd ' + path_tmp + 'EVAL ; rm ms2raster.py')
        os.system('cd ' + path_tmp + 'EVAL ; rm mssel')
        os.system('cd ' + path_tmp + 'EVAL ; rm seedms')
        os.system('cd ' + path_tmp + 'EVAL ; rm stepftn')
        os.system('cd ' + path_tmp + 'EVAL ; rm tp.out')
        os.system('cd ' + path_tmp + 'EVAL ; rm trajfixconst')
        os.system('cd ' + path_tmp + 'EVAL ; rm -r __MACOSX')
    

# os.system('clear')
# path = os.getcwd() + '/'
# print('Path corrente: ' + path)

# mod = input('\nAvviare in modalità SELECTION[S], NEUTRAL[N] o entrambe[B]?: ')
# create_tree(path, mod)


# clean_tree(path, mod)


