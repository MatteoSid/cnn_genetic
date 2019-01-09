# image_generator_modules

Lo script **image_generator.py** usa il modulo **dir_restore.py** per preparare le cartelle che conterranno tutti i dati del dataset divisi per tipo mentre il modulo **matrix_to_image.py** viene usato all’interno dello script per convertire le singole matrici in file .png.

**image_generator.py** crea le immagini con questa sequenza di istruzioni:
1. Il modulo **dir_restore.py** controlla se è presente la struttura di cartelle necessaria al salvataggio dei file. Il modulo procede in questo modo:
* Se è presente la ripulisco in modo da avere le cartelle vuote;
* Se non è presente la creo.
2. Usa dei comandi bash per eseguire **ms.c** e creare i file .txt necessari alla creazione di matrici. Quesi file saranno salvati tutti nella cartella `/dataset/1_original_dataset` creata in precedenza;
3. Ogni volta che crea una matrice la salvo in un file .txt e contemporaneamente crea anche il corrispettivo file .txt senza le info non necessarie e lo salvo in `/dataset/2_reshaped_dataset`;
4. Una volta create le matrici di soli zeri e uni la trasformo in immagine **.png** usando **matrix_to_image_fn.py** e salvo le singole immagini in `/dataset/3_image_dataset`.