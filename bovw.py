import numpy as np
from skimage import io as sio
from skimage.feature import daisy
from time import time
from matplotlib import pyplot as plt
from skimage.color import rgb2grey

def extract_features(dataset):
    #ottieni il numero totale di immagini nel dataset
    nimgs = dataset.getLength()
    #crea una lista vuota di features
    features = list()
    ni=0 #numero di immagini analizzate finora
    total_time=0
    for cl in dataset.getClasses():
        paths=dataset.paths[cl]
        for impath in paths:
            t1=time() #timestamp attuale
            im=sio.imread(impath,as_grey=True) #carica immagine in scala di grigi
            feats=daisy(im) #estrai features
            feats=feats.reshape((-1,200)) #reshape dell'array'
            features.append(feats) #aggiungi features alla lista
            t2=time() #timestamp attuale
            t3=t2-t1 #tempo trascorso
            total_time+=t3
            #Stampa un messaggio di avanzamento, con la stima del tempo rimanente
            ni+=1 #aggiorna il numero di immagini analizzate finora
            if nimgs-ni==5:
                print ("...")
            if nimgs-ni<5:
                print ("Image {0}/{1} [{2:0.2f}/{3:0.2f} sec]".format(ni,nimgs,t3,t3*(nimgs-ni)))
    print ("Stacking all features...")
    t1=time()
    stacked = np.vstack(features) #metti insieme le feature estratte da tutte le immagini
    t2=time()
    total_time+=t2-t2
    print ("Total time: {0:0.2f} sec".format(total_time))
    return stacked

def extract_and_describe(img,kmeans):
    #estrai le feature da una immagine
    features=daisy(rgb2grey(img)).reshape((-1,200))
    #assegna le feature locali alle parole del vocabolario
    assignments=kmeans.predict(features)
    #calcola l'istogramma
    histogram,_=np.histogram(assignments,bins=500,range=(0,499))
    #restituisci l'istogramma normalizzato
    return histogram

def display_image_and_representation(X,y,paths,classes,i):
	im=sio.imread(paths[i])
	plt.figure(figsize=(12,4))
	plt.suptitle("Class: {0} - Image: {1}".format(classes[y[i]],i))
	plt.subplot(1,2,1)
	plt.imshow(im)
	plt.subplot(1,2,2)
	plt.plot(X[i])
	plt.show()

def show_image_and_representation(img,image_representation):
    plt.figure(figsize=(13,4))
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.plot(image_representation)
    plt.show()
	
def compare_representations(r1,r2):
	plt.figure(figsize=(12,4))
	plt.subplot(1,2,1)
	plt.plot(r1)
	plt.subplot(1,2,2)
	plt.plot(r2)
	plt.show()

def describe_dataset(dataset,kmeans):
    y=list() #inizializziamo la lista delle etichette
    X=list() #inizializziamo la lista delle osservazioni
    paths=list() #inizializziamo la lista dei path
    
    classes=dataset.getClasses()
    
    ni=0
    t1=time()
    for cl in classes: #per ogni classe
        for path in dataset.paths[cl]: #per ogni path relativo alla classe corrente
            img=sio.imread(path,as_grey=True) #leggi imamgine
            feat=extract_and_describe(img,kmeans) #estrai features
            X.append(feat) #inserisci feature in X
            y.append(classes.index(cl)) #inserisci l'indice della classe corrente in y
            paths.append(path) #inserisci il path dell'immagine corrente alla lista
            ni+=1
            #rimuovere il commento di seguito per mostrare dei messaggi durante l'esecuzione
            #print "Processing Image {0}/{1}".format(ni,total_number_of_images)

    #Adesso X e y sono due liste, convertiamole in array
    X=np.array(X)
    y=np.array(y)
    t2=time()
    print ("Elapsed time {0:0.2f}".format(t2-t1))
    return X,y,paths