# image_generator_modules

Lo script **image_generator.py** usa il modulo **dir_restore.py** per preparare le cartelle che conterranno tutti i dati del dataset divisi per tipo mentre il modulo **matrix_to_image.py** viene usato all’interno dello script per convertire le singole matrici in file .png.

**image_generator.py** crea le immagini con questa sequenza di istruzioni:
1. Controlla se è presente la struttura di cartelle necessaria al salvataggio dei file:
* Se è presente la ripulisco in modo da avere le cartelle vuote;
* Se non è presente la creo.
Il modulo che si occupa di preparare le cartelle è **dir_restore.py**
2. Usa dei comandi bash per eseguire **ms.c** e creare i file .txt necessari alla creazione di matrici. Quesi file saranno salvati tutti nella cartella /dataset/11_original_dataset;
3. Ogni volta che crea una matrice in un file .txt crea il corrispettivo file .txt senza le info non necessarie e lo salvo in /dataset/2_reshaped_dataset;
4. Una volta create le matrici di soli zeri e uni la trasformo in immagine usando **matrix_to_image_fn.py**
