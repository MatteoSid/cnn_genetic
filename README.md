# cnn_mnist_beta#.py
I file e le cartelle principali sono:
* **cnn_mnist_beta#.py** è la funzione principale di riferimento;
* **cnn_mnist_fn.py** contiene la funzione che crea il modello (usata nelle vecchie versioni, deve ancora essere implementata nell'ultima versine);
* **dataset_creator** contiene uno script che usa ms2raster.py ed altri moduli per creare dataset di immagini già classificate e divise per tipo per eseguire training e test della rete.
* **MNIST_data** è una cartella che contiene i dataset di training e testing.

---
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
