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

# Gaussian Mixture Model

## come funziona

Il Gaussian Mixture Model è un modello di clustering che utilizza una distribuzione gaussiana per modellare ogni
cluster.

## come lo implementiamo

Per implementare il GMM abbiamo utilizzato la libreria sklearn, che ci ha permesso di utilizzare il metodo GMM.
Il metodo GMM ci ha permesso di creare un modello che ha 10 cluster, cioè 20 classi.
Il metodo GMM ci ha permesso di fare il fit del modello, cioè di addestrare il modello con i dati di training.
Il metodo GMM ci ha permesso di fare il predict del modello, cioè di predire le classi dei dati di test.
