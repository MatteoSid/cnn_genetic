# dataset_creator
La cartella `dataset_creator` contiene lo script **matrix_to_image.py** che utilizza lo script **ms2raster.py** per generare le metrici sottoforma di file di testo con matrici di zeri e uni e li converte in immagini. Per la generazione delle matrici sootoforma di file di testo si usa **ms2raster.py** che a sua volta usa un output modificato di **ms**. Oltre a questi script principali sono presenti anche tutti i moduli per il corretto funzionamento.

## dataset_creator.py
Per prima cosa elimina ogni traccia di file `*.sim` ovvero ogni file del dataset (da togliere). Dopodiché si ricava il path della cartella di esecuzione che servirà per muoversi all'interno di tutte le cartelle che verranno generate in seguito e per il corretto funzionamento di **ms2raster.py** che richiede il path corrente come input. Vengono poi chiesti tutti i parametri in input:
* Quale dataset creare tra `SELECTION [S]`, `NEUTRAL [N]` O entrambi `[B]`;
* Grandezza delle singole matrici;
* Numero di matrici da generare rispettivamente per il `TRAINING SET` e per il `TEST SET`;
* Intensità di selezione.

Prima di cominciare a generare le matrici preparo i file per la creazione dei dataset e la struttura di cartelle che andrà ad ospitare i vari dataset divisi per tipo. Per farlo uso la funzione **create_tree()** contenuta nel modulo **foldertreemng.py** che in caso non ci fosse nessuna struttura di cartelle presenti ne creerebbe una mentre se trovasse residui di esecuzioni precedenti le andrebbe ad eliminare per ricreare la struttura necessaria con cartelle con all'interno solo i file necessari.
Una volta raccolti i parametri necessari e create le cartelle vengono eseguiti i comandi con le giuste combinazioni di comandi a seconda dei parametri inseriti.
Una volta creato il seti di matrici in formato `*.sim` viene chiesto all'utente se si desidera trasformarlo in immagini `*.png`.
In caso affermativo vengono create le rispettive cartelle e viene usata la funzione **converter_fn()** contenuta nel modulo **matrix_to_image_fn.py** per trasformare ogni matrice in immagine.

Alla fine di tutto il processo usiamo la funzione **clean_tree()**, sempre contenuta all'interno del modulo **foldertreemng.py**, per eliminare tutti i file che non servono più o che sono presenti più volte inutilmente.

**-> I DATI VENGONO SOVRASCRITTI AD OGNI ESECUZIONE <-**

## foldertreemng.py
Il modulo **foldertreemng.py** contiene due funzioni per la gestione dei file e delle cartelle necessari al corretto funzionamento del programma. 
* **create_tree(path, mode)** in base alla modalità selezinoata crea la struttura di cartelle con all'interno i file necessari per l'esecuzione di **ms2raster.py** presi dal file `ms2raster.zip` presente nella cartella principale che viene estratto in ogni ultima sottocartella creata. Inoltre, all'interno della funzione, è presente anche un controllo che elimina tutti i file e le cartelle delle vecchie esecuzioni e prepara le cartelle e i file per l'esecuzione corrente.
* **clean_tree(path, mode)** ripulisce le cartelle contenenti il dataset dai file non più necessari che erano stati estratti in precedenza da `ms2raster.zip`.

## matrix_to_image_fn.py
Contiene la funzione **converter_fn(path, file_name)** che converte le matrici generate in immagini `*.png`. Per farlo prende in input il percorso delle matrici in formato `*.sim` e il nome con cui voglio salvare queste immagini (path completo + nome del file).