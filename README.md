# cnn_mnist_beta4.py
I file e le cartelle principali sono:
* **cnn_mnist_*.py** è la funzione principale di riferimento;
* **cnn_mnist_fn.py** contiene la funzione che crea il modello (usata nelle vecchie versioni, deve ancora essere implementata nell'ultima versine);
* **msdir** contiene tutti i file relativi allo script `ms` per generare simulazioni di dati genetici (versione da aggiornare);
* **image_generator.py** è uno script che usa i moduli contenuti in `/image_generator_modules` per creare immagini ti training;
* **tqdm** contiene dei moduli per creare barre di caricamento;
* **MNIST_data** è una cartella che contiene i dataset di training e testing.

## cnn_mnist_*.py

È la funzione principale per l'esecuzione del modello. All'avvio per prima cosa richiede l'inserimento della modalità di avvio che può essere `TRAIN`, `TEST`, `BOTH`.

* Modalità `TRAIN`: Per ora il programma chiede solamente il nunmero di epoche come parametro di input, gli altri parametri sono costanti all'interno del codice. Una volta inserito il numero di epoche verrà avviato il training dell'allenamento che darò info in modo periodico sull'avanzamento dell'allenamento. Verrà anche segnalato con una checkbox quando l'accuratezza della rete è migliorata. Per esempio, se alleniamo il modello con 50 epoche avremo un output del tipo:

```
TRAINING MODE
16      0.9141          [X]
20      0.8516          [ ]
22      0.9141          [X]
31      0.9375          [X]
40      0.8984          [ ]
41      0.9375          [X]
42      0.9531          [X]
```
con log ogni 20 epoche oppure ogni volta che l'accuratezza è migliorata.

* Modalità `TEST`: Anche questa volta verrà chiesto solamente il numero di epoche da eseguire, solo che questa volta il modellolavorerà sul dataset di test e l'output darà solamente informazioni sull'accuratezza del modello durante il test. Per esempio, se testiamo il modello con 50 epoche avremo un output del tipo:

```
TESTING MODE
Initialization Complete
Testing Accuracy:0.9453125
Testing finished
```
* Modalità `BOTH`: Esegue entrambe le modalità facendo prima il `TRAIN` e poi il `TEST`.


## image_generator 
All'avvio prende in input le seguenti informazioni con cui è possibile gestire in che modo verranno generate le matrici:
* Numero di matrici da generare;
* Dimensione delle singole matrici;
* Valore del parametro Delta.

Lo script **image_generator.py** usa il modulo **dir_restore.py** per preparare le cartelle che conterranno tutti i dati del dataset divisi per tipo mentre il modulo **matrix_to_image.py** viene usato all’interno dello script per convertire le singole matrici in file .png.

**image_generator.py** crea le immagini con questa sequenza di istruzioni:
1. Il modulo **dir_restore.py** controlla se è presente la struttura di cartelle necessaria al salvataggio dei file. Il modulo procede in questo modo:
* Se è presente la ripulisco in modo da avere le cartelle vuote;
* Se non è presente la creo.
2. Usa dei comandi bash per eseguire **ms.c** e creare i file .txt necessari alla creazione di matrici. Quesi file saranno salvati tutti nella cartella `/dataset/1_original_dataset` creata in precedenza;
3. Ogni volta che crea una matrice la salvo in un file .txt e contemporaneamente crea anche il corrispettivo file .txt senza le info non necessarie e lo salvo in `/dataset/2_reshaped_dataset`;
4. Una volta create le matrici di soli zeri e uni la trasformo in immagine **.png** usando **matrix_to_image_fn.py** e salvo le singole immagini in `/dataset/3_image_dataset`.
