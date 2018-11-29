# cnn_mnist_beta4.py

* **cnn_mnist_*.py** è il file principale di riferimento
* **cnn_mnist_fn.py** contiene la funzione che crea il modello
* **msdir** contiene tutti i file relativi allo script ms per generare simulazioni di dati genetici (versione da aggiornare)
* **image_generator.py** è uno script che usa i moduli contenuti in `/image_generator_modules`
* **tqdm** contiene dei moduli per creare barre di caricamento
* **MNIST_data** contiene i dataset di trining e testing

**cnn_mnist_*.py** è la funzione principale per l'esecuzione del modello. All'avvio per prima cosa richiede l'inserimento della modalità di avvio che può essere `TRAIN`, `TEST`, `BOTH`.

1. Modalità `TRAIN`: Per ora il programma chiede solamente il nunmero di epoche come parametro di input, gli altri parametri sono costanti all'interno del codice. Una volta inserito il numero di epoche verrà avviato il training dell'allenamento che darò info in modo periodico sull'avanzamento dell'allenamento. Verrà anche segnalato con una checkbox quando l'accuratezza della rete è migliorata. Per esempio, se alleniamo il modello con 50 epoche avremo un output del tipo:

```
TRAINING MODE
20      0.8828          [X]
23      0.9141          [X]
28      0.9531          [X]
39      0.9531          [X]
40      0.8984          [ ]
44      0.9531          [X]
45      0.9531          [X]
```
con log ogni 20 epoche oppure ogni volta che l'accuratezza è migliorata.

2. Modalità `TEST`: Anche questa volta verrà chiesto solamente il numero di epoche da eseguire, solo che questa volta il modellolavorerà sul dataset di test e l'output darà solamente informazioni sull'accuratezza del modello durante il test. Per esempio, se testiamo il modello con 50 epoche avremo un output del tipo:
```
TESTING MODE
Initialization Complete
Testing Accuracy:0.9453125
Testing finished
```

**image_generator** all'avvio prende in input le seguenti informazioni con cui è possibile gestire in che modo verranno generate le matrici:
* Numero di matrici da generare;
* Dimensione delle singole matrici;
* Valore del parametro Delta.
