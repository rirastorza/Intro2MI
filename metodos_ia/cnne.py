#Red Convolucional que estima el radio, centroide (x,y)
#y propiedades dielectricas (eps,sigma) de un cilindro infinito
#en Z (2D) a partir de la lectura de magnitud de
#campo en 16 antenas receptoras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

import numpy as np
import random
from matplotlib import pyplot


#Variables y Funciones
numdat = 9000 #Numero de datos a cargar
nant = 8 #Cantidad de antenas con que se hace la medición
Einc = np.zeros((1,nant**2,2))
Data = np.zeros((numdat,nant**2,2))
Param = np.zeros((numdat,5))
Emnorm = np.zeros((numdat,nant**2)) #Modulo de Et/Einc
Epnorm = np.zeros((numdat,nant**2)) #Fase como faseunwrapped(Et)-faseunwrapped(Einc)

def escalae(rawpoints, high=1.0, low=0.0):
    mins = 0.00
    maxs = 15.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalap(rawpoints, high=1.0, low=0.0):
    mins = -10.50
    maxs = 10.50
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalar(rawpoints, high=1.0, low=0.0):
    mins = 2.0e-3
    maxs = 4.0e-2
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalacoord(rawpoints, high=1.0, low=0.0):
    mins = -7.0e-2
    maxs = 7.0e-2
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalaeps(rawpoints, high=1.0, low=0.0):
    mins = 8.0
    maxs = 82.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalasigma(rawpoints, high=1.0, low=0.0):
    mins = 0.3
    maxs = 1.70
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainve(rawpoints, high=15.0, low=0.0):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainvp(rawpoints, high=10.50, low=-10.50):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainvr(rawpoints, high=4.0e-2, low=2.0e-3):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainvcoord(rawpoints, high=7.0e-2, low=-7.0e-2):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainveps(rawpoints, high=82.0, low=8.0):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def escalainvsigma(rawpoints, high=1.70, low=0.3):
    mins = 0.0
    maxs = 1.0
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)


#Levantamiento de Data 
trnoise = 0.05
for i in range(numdat):
    dataset = np.loadtxt('AllData/Data'+str(i+1)+'.out', dtype='str', delimiter="\n", comments='#')
    for j in range(256):
        temp = dataset[j].split()
        Data[i][j][0] = np.random.normal(float(temp[2]), trnoise*float(temp[2]))
        Data[i][j][1] = np.random.normal(float(temp[3]), trnoise*float(temp[3]))

dataset = np.loadtxt('AllData/Einc.out', dtype='str', delimiter="\n", comments='#')
for j in range(256):
    temp = dataset[j].split()
    Einc[0][j][0] = np.random.normal(float(temp[2]), trnoise*float(temp[2]))
    Einc[0][j][1] = np.random.normal(float(temp[3]), trnoise*float(temp[3]))

for i in range(numdat):
    dataset = np.loadtxt('AllData/Param'+str(i+1)+'.out', dtype='str', delimiter="\n", comments='#')
    for j in range(5):
        Param[i][j] = float(dataset[j])

for i in range(numdat):
    for j in range(256):
        Emnorm[i][j] = Data[i][j][0]/Einc[0][j][0]
        #Epnorm[i][j] = Data[i][j][1]-Einc[0][j][1]


Emnorm = np.reshape(Emnorm,(numdat,16,16,1))

# Normalizacion de Entrada y Salida de la Red Neuronal
Emnorm = escalae(Emnorm) 
#Epnorm = escalap(Epnorm)
Param[:,0] = escalar(Param[:,0])
Param[:,1] = escalacoord(Param[:,1])
Param[:,2] = escalacoord(Param[:,2])
Param[:,3] = escalaeps(Param[:,3])
Param[:,4] = escalasigma(Param[:,4])

'''
picture = []
fig, ax = pyplot.subplots()
image = Emnorm[4]
print(len(image[0]))
for i in range(len(image)):
    for j in range (len(image[0])):
        picture.append(image[i][j])
picture = np.asarray(picture)
picture = np.reshape(picture,(16,16))
ax.imshow(picture, cmap=pyplot.cm.gray, interpolation='nearest')
ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 16))
ax.spines['bottom'].set_position(('outward', 16))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

pyplot.show()
'''

# Data de entrenamiento y de validacion
numtrain = 8000 #nro de datos de entrenamiento
numval = numdat - numtrain
DataTrain = Emnorm#np.concatenate((Emnorm, Epnorm), axis=1)

X = DataTrain[0:numtrain , :]
Y = Param[0:numtrain , 0:5]
Xval = DataTrain[numtrain+1:numdat , :]
Yval = Param[numtrain+1:numdat , 0:5]

# Modelo de CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(16,16,1)))	
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2304, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='linear'))

# Compile model
sgd = optimizers.Adam(lr=0.01)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse','mae'])#'mse', 'mae', 'mape', 'cosine''mean_squared_error'
# Fit the model
history = model.fit(X, Y, validation_data=(Xval, Yval), epochs=500, batch_size=50)
# evaluate the model
scores = model.evaluate(Xval, Yval, batch_size=50)
pred = model.predict(Xval)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

print('MAE Radio: ',np.average(abs(escalainvr(pred[:,0])-escalainvr(Yval[:,0]))))
print('MAE Xc: ',np.average(abs(escalainvcoord(pred[:,1])-escalainvcoord(Yval[:,1]))))
print('MAE Yc: ',np.average(abs(escalainvcoord(pred[:,2])-escalainvcoord(Yval[:,2]))))
print('MAE Eps: ',np.average(abs(escalainveps(pred[:,3])-escalainveps(Yval[:,3]))/escalainveps(Yval[:,3])))
print('MAE Sigma: ',np.average(abs(escalainvsigma(pred[:,4])-escalainvsigma(Yval[:,4]))/escalainvsigma(Yval[:,4])))


#Save model to JSON
model_json = model.to_json()
with open("Redes/cnn_AmplitudeOnly2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Redes/cnn_AmplitudeOnly2.h5")
print("Saved model to disk")


#print(history.history.keys())
# plot metrics
pyplot.figure()
pyplot.plot(escalainvr(pred[:,0]),escalainvr(Yval[:,0]),'ok', markersize = 0.5, alpha = 0.4,label = 'Radius')
pyplot.ylim((1.0e-3,4.0e-2))
pyplot.xlim((1.0e-3,4.0e-2))
pyplot.xlabel('Valor Predicho')
pyplot.ylabel('Valor Real')
pyplot.legend(loc='lower right', shadow='False')

pyplot.figure()
pyplot.plot(escalainvcoord(pred[:,1]),escalainvcoord(Yval[:,1]),'ok', markersize = 0.5, alpha = 0.4, label = 'Xcen')
pyplot.ylim((-7.0e-2,7.0e-2))
pyplot.xlim((-7.0e-2,7.0e-2))
pyplot.xlabel('Valor Predicho')
pyplot.ylabel('Valor Real')
pyplot.legend(loc='lower right', shadow='False')

pyplot.figure()
pyplot.plot(escalainvcoord(pred[:,2]),escalainvcoord(Yval[:,2]),'ok', markersize = 0.5, alpha = 0.4, label = 'Ycen')
pyplot.ylim((-7.0e-2,7.0e-2))
pyplot.xlim((-7.0e-2,7.0e-2))
pyplot.xlabel('Valor Predicho')
pyplot.ylabel('Valor Real')
pyplot.legend(loc='lower right', shadow='False')

pyplot.figure()
pyplot.plot(escalainveps(pred[:,3]),escalainveps(Yval[:,3]),'ok', markersize = 0.5, alpha = 0.4, label = '\epsilon')
pyplot.ylim((5.0,85.0))
pyplot.xlim((5.0,85.0))
pyplot.xlabel('Valor Predicho')
pyplot.ylabel('Valor Real')
pyplot.legend(loc='lower right', shadow='False')

pyplot.figure()
pyplot.plot(escalainvsigma(pred[:,4]),escalainvsigma(Yval[:,4]),'ok', markersize = 0.5, alpha = 0.4, label = '\sigma')
pyplot.ylim((0.2,1.8))
pyplot.xlim((0.2,1.8))
pyplot.xlabel('Valor Predicho')
pyplot.ylabel('Valor Real')
pyplot.legend(loc='lower right', shadow='False')

pyplot.figure()
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()


