# Clustering

Terzo assignment di Informatica 2. il main topic è il clustering.

## clusterings

Il clustering è un metodo di analisi dei dati non supervisionato, che permette quindi di raggruppare insiemi di dati in
base
alle loro caratteristiche.
In quanto non supervisionato, non necessita di un insieme di dati di training, ma solo dati di input, che non sono
etichettati
e non sono presenti informazioni a priori su quali dati devono essere raggruppati insieme.
Il risultato di un clustering è un insieme di cluster ognuno dei quali è formato da serie di dati simili tra loro.

## il dataset che andiamo ad utilizzare è sameion.csv

Il dataset che andiamo ad analizzare è semeion.csv, che contiene 1593 immagini di 16x16 pixel, che rappresentano numeri
scritti a mano.
Il dataset è composto da 1593 righe, ognuna delle quali rappresenta un'immagine, e 256 colonne, ognuna delle quali
rappresenta un pixel.
Le ultime 10 colonne rappresentano che numero è rappresentato dall’immagine, questa informazione non verrà utilizzata in
fase di clustering, ma sarà utilizzata successivamente, nel momento in cui andremo a valutare le performance dei vari
metodi utilizzati.  
Le classi sono numeri da 0 a 9, che rappresentano i numeri scritti a mano.

## Principal Compoment Analysis (PCA)

PCA è un metodo di analisi dei dati che permette di ridurre la dimensionalità dei dati, cioè di ridurre il numero di
caratteristiche che descrivono i dati.
Un vantaggio di questo metodo è che permette di ridurre il tempo di calcolo, in quanto riduce il numero di operazioni
che devono essere eseguite.
Un altro vantaggio è che permette di visualizzare i dati in un grafico bidimensionale, in quanto riduce la
dimensionalità
dei dati da 256 a 2.
Un altro vantaggio è che permette di visualizzare i dati in un grafico bidimensionale, in quanto riduce la
dimensionalità
dei dati da 256 a 2.
PCA funziona in questo modo:

- calcola la media di tutti i dati
- calcola la matrice di covarianza
- calcola gli autovalori e gli autovettori della matrice di covarianza
- seleziona gli autovettori con gli autovalori più grandi
- calcola la matrice di proiezione dei dati
- calcola i dati proiettati

#  