# cnn_genetic.py
I file e le cartelle principali sono:
* **cnn_genetic.py** è la funzione principale di riferimento;
* **cnn_model_fn.py** contiene la funzione che crea il modello;
* **load_dataset.py** è una funzione per il caricamento in memoria del dataset.

---
## cnn_genetic.py

È la funzione principale per l'esecuzione del modello. All'avvio per prima cosa richiede l'inserimento della modalità di avvio che può essere `TRAIN`, `TEST`, `BOTH`.

* Modalità `TRAIN`: Il programma chiede in input la grandezza del parametro `batch_size` e il numero di `epoche` che si vogliono eseguire. 
Una volta avviato il training verrà allenato il modello sull'intero dataset per ogni epoca e per ogni epoca verranno fatte tante iterazioni quante ne servono per arrivare alla fine del dataset, quindi il numero i iterazioni per ogni epoca dipende da `batch_size` (maggiore è il valore di `batch_size` e minori saranno le iterazioni necessarie e viceversa). Alla fine di ogni epoca, dopo l'ultima iterazione, verrà fatto un `validation test` su immagini che il modello elabora per la prima volta in modo da verificare l'**accuracy** raggiunta. 

Durante il training verrano date informazioni in modo continuo sull'avanzamento dell'allenamento. Verrà anche segnalato con una checkbox quando l'accuratezza della rete è migliorata e alla fine di ogni epoca verrà visualizzato il risultato del `validation test`. 

```
TRAINING MODE
...
...
[e:4, i:79]		0.9570		[ ]		5.423s
[e:4, i:80]		0.9629		[ ]		5.300s
[e:4, i:81]		0.9648		[X]		5.193s
[e:4, i:82]		0.9668		[X]		5.193s
[e:4, i:83]		0.9609		[ ]		5.316s
[e:4, i:84]		0.9629		[ ]		5.298s
[e:4, i:85]		0.9727		[X]		5.277s
[e:4, i:86]		0.9707		[ ]		5.305s
Evaluation test: 0.9628906
```


* Modalità `TEST`: Il programma caricherà il modello ed eseguirà il test sul dataset di test che contiene immagini che non sono mai state usate per l'allenamento. 

```
TESTING MODE
Initialization Complete
Testing Accuracy:0.9453125
Testing finished
```
* Modalità `BOTH`: Esegue entrambe le modalità facendo prima il `TRAIN` e poi il `TEST`.

---

# load_dataset.py
È un modulo che contiene una funzione **get_images()** per caricare il dataset e una **next_batch()** che lo divide in batch per poi darli come input alla rete neurale. 

## get_images(files_path, img_size_w, img_size_h, mode, randomize)
La funzione prende in input:
* `files_path`: una stringa che indica il percorso della cartella DATASET creata da **dataset_creator.py**;
* `img_size_h`, `img_size_w`: due vaori che esprimono rispettivamente l'altezza e la larghezza delle immagini da caricare;
* `mode`: è la modalità in cui si sta lavorando, se **mode='TRAIN'** allora verrà caricato il dataset di training mentre se **mode='TEST'** verrà caricato il dataset di test.
* `randomize`: parametro settato a **False** di default. Se impostato a **true** mescola l'array delle immagini e quello delle labels con lo stesso ordine random così entrambi gli array avranno un ordine casuale ma senza perdere la correlazione con le rispettive labels. Se si lascia impostato a **False** come di default avremo un dataset in cui la prima metà contiene le immagini relative a **SELECTION** e la seconda metà conterrà le immagini relative a **NEUTRAL**.

Il dataset viene caricato in due array:
* `images_arr`: per le immagini;
* `label_arr`: per le rispettive etichette.

Viene caricato prima il **SELECTOIN** dataset, ogni immagine caricata viene messa in **images_arr** e per ogni immagine caricata viene inserita un'etichetta **[0,1]** nel **label_arr**. 
Viene fatta la stessa cosa anche per il **NEUTRAL** datast ma in questo caso le etichette inserite in **label_arr** sono **[1,0]**.
Una volta inserite tutte le imagini in **images_arr** viene fatto un *reshape* per fare in modo che ad ogni riga dell'array corrisponda un'inera immagine.
La funzione ritorna `len(images_arr)`, `images_arr`, `label_arr`.

```
Script avviato in modalità TRAIN

Verranno usati i seguenti dataset:
/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/SELECTION/TRAIN_IMG/
/Users/matteo/Documents/GitHub/Cnn_Genetic/cnn_genetic/DATASET/NEUTRAL/TRAIN_IMG/

Carico il SELECTION dataset:
100%|█████████████████████████████████████████████████████| 4877/4877 [00:16<00:00, 302.93it/s]

Carico il NEUTRAL dataset:
100%|█████████████████████████████████████████████████████| 5000/5000 [00:15<00:00, 314.04it/s]

Gli array hanno dimensione:
  -images_arr: 9877
  -label_arr: 9877

images_array shape: (9877, 1000, 48)
label_array shape: (9877, 2)
images_arr reshaped: (9877, 48000)
```

**log:** inoltre all'interno della directory `DATASET` viene salvato un file **log.txt** con un log sull'esecuzione dello script. 
**NOTA:** c'è anche la possibilità di salvare l'array numpy su disco come file `.npy` (commentata).

# next_batch(total, images, labels, batch_size, index)
È una funzine che, dato un array di immagini e uno di labels, li divide in batch per poterli passare alla rete neurale. La funzione prende in input:
* `total`: un **int** che indica la lunghezza totale del dataset quindi il varlore corrispondente al **len(images_arr)** ritornato dal **get_images()**;
* `images`: array contenente le immagini;
* `labels`: array contenente tutte le labels;
* `batch_size`: un **int** che indica quanto grandi devono essere i batch da creare;
* `index`: un **int** che indica l'indice a cui sono arrivato a dividere l'intero dataset.
