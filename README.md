# cnn_mnist_beta4.py

* **cnn_mnist_*.py** è il file principale di riferimento.
* **cnn_mnist_fn.py** contiene la funzione che crea il modello
* **image_generator.py** è uno script che usa i moduli contenuti in **image_generator_modules**
* **tqdm** contiene dei moduli per creare barre di caricamento
* **MNIST_data** contiene i dataset di trining e testing


Lo script **image_generator.py** usa il modulo **dir_restore.py** per preparare le cartelle che conterranno tutti i dati del dataset divisi per tipo mentre il modulo **matrix_to_image.py** viene usato all’interno dello script per convertire le singole matrici in file .png.
