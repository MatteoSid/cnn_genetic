import os

# os.system('clear')
# path = os.getcwd() + '/'
# print('Path corrente: ' + path)

# mod = input('\nAvviare in modalità SELECTION[S] o NEUTRAL[N]? ')

def create_tree(path, mode):
    if mode == 'S' or mode == 'B':
            if os.access(path + 'SELECTION', os.F_OK) == True:
                os.system('rm -r ' + path + 'SELECTION')
                os.mkdir(path + 'SELECTION')
                os.mkdir(path + 'SELECTION/TRAIN/')
                os.mkdir(path + 'SELECTION/TEST/')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'SELECTION/TRAIN')
                os.system('cd ' + path + 'SELECTION/TRAIN' + ' ; unzip ms2raster.zip')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'SELECTION/TEST')
                os.system('cd ' + path + 'SELECTION/TEST' + ' ; unzip ms2raster.zip')
            else:
                os.mkdir(path + 'SELECTION')
                os.mkdir(path + 'SELECTION/TRAIN/')
                os.mkdir(path + 'SELECTION/TEST/')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'SELECTION/TRAIN')
                os.system('cd ' + path + 'SELECTION/TRAIN' + ' ; unzip ms2raster.zip')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'SELECTION/TEST')
                os.system('cd ' + path + 'SELECTION/TEST' + ' ; unzip ms2raster.zip')

    if mode == 'N' or mode == 'B':
            if os.access(path + 'NEUTRAL', os.F_OK) == True:
                os.system('rm -r ' + path + 'NEUTRAL')
                os.mkdir(path + 'NEUTRAL')
                os.mkdir(path + 'NEUTRAL/TRAIN/')
                os.mkdir(path + 'NEUTRAL/TEST/')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'NEUTRAL/TRAIN')
                os.system('cd ' + path + 'NEUTRAL/TRAIN' + ' ; unzip ms2raster.zip')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'NEUTRAL/TEST')
                os.system('cd ' + path + 'NEUTRAL/TEST' + ' ; unzip ms2raster.zip')
            else:
                os.mkdir(path + 'NEUTRAL')
                os.mkdir(path + 'NEUTRAL/TRAIN/')
                os.mkdir(path + 'NEUTRAL/TEST/')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'NEUTRAL/TRAIN')
                os.system('cd ' + path + 'NEUTRAL/TRAIN' + ' ; unzip ms2raster.zip')
                os.system('cp ' + path + 'ms2raster.zip ' + path + 'NEUTRAL/TEST')
                os.system('cd ' + path + 'NEUTRAL/TEST' + ' ; unzip ms2raster.zip')

def clean_tree(path, mode):
    if mode == 'S' or mode =='B':
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm ms2raster.zip')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm ms')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm lastp0')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm ms2raster.py')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm mssel')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm seedms')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm stepftn')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm tp.out')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm trajfixconst')
        os.system('cd ' + path + 'SELECTION/TRAIN ; rm -r __MACOSX')

        os.system('cd ' + path + 'SELECTION/TEST ; rm ms2raster.zip')
        os.system('cd ' + path + 'SELECTION/TEST ; rm ms')
        os.system('cd ' + path + 'SELECTION/TEST ; rm lastp0')
        os.system('cd ' + path + 'SELECTION/TEST ; rm ms2raster.py')
        os.system('cd ' + path + 'SELECTION/TEST ; rm mssel')
        os.system('cd ' + path + 'SELECTION/TEST ; rm seedms')
        os.system('cd ' + path + 'SELECTION/TEST ; rm stepftn')
        os.system('cd ' + path + 'SELECTION/TEST ; rm tp.out')
        os.system('cd ' + path + 'SELECTION/TEST ; rm trajfixconst')
        os.system('cd ' + path + 'SELECTION/TEST ; rm -r __MACOSX')
    
    if mode == 'N' or mode == 'B':
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm ms2raster.zip')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm ms')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm lastp0')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm ms2raster.py')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm mssel')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm seedms')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm stepftn')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm tp.out')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm trajfixconst')
        os.system('cd ' + path + 'NEUTRAL/TRAIN ; rm -r __MACOSX')

        os.system('cd ' + path + 'NEUTRAL/TEST ; rm ms2raster.zip')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm ms')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm lastp0')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm ms2raster.py')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm mssel')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm seedms')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm stepftn')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm tp.out')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm trajfixconst')
        os.system('cd ' + path + 'NEUTRAL/TEST ; rm -r __MACOSX')
    

    

# create_tree(path, mod)
# clean_tree(path + 'SELECTION/')

