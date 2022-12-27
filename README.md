# Clustering
Terzo assignment di Informatica 2. il main topic è il clustering.

## clusterings in generale
Un clustering è un metodo di analisi dei dati che permette di raggruppare insiemi di dati in base alle loro caratteristiche. 
Il clustering è un metodo di apprendimento non supervisionato, in quanto non richiede dati di training, ma solo dati di input.

## il dataset che andiamo ad utilizzare è sameion.csv
Il dataset è composto da 6000 righe e 256 colonne. Le prime 256 colonne sono le features, le ultime 10 sono le classi.
ogni riga rappresenta un'immagine di dimensione 16x16 pixel, ogni pixel è rappresentato da un valore 0 se è bianco, 1 se è nero.
ogni immagine è rappresentata da 256 features, che sono i valori dei pixel.
ogni immagine è rappresentata da 10 classi, che sono i valori delle lettere che rappresentano l'immagine.

# Gaussian Mixture Model
## come funziona
Il Gaussian Mixture Model è un modello di clustering che utilizza una distribuzione gaussiana per modellare ogni cluster.

## come lo implementiamo
Per implementare il GMM abbiamo utilizzato la libreria sklearn, che ci ha permesso di utilizzare il metodo GMM.
Il metodo GMM ci ha permesso di creare un modello che ha 10 cluster, cioè 20 classi.
Il metodo GMM ci ha permesso di fare il fit del modello, cioè di addestrare il modello con i dati di training.
Il metodo GMM ci ha permesso di fare il predict del modello, cioè di predire le classi dei dati di test.
