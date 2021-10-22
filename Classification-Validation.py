import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook

from numpy.random import seed
from numpy.random import shuffle
from numpy.random import randint


#Cargar datos
data = np.load("datos/clasificacion_p1.npy")

#Extraer informacion
x = data[0:2, :]
y = data[2, :]

#Ordenar informacion
x = np.transpose(x)
y = y[:, np.newaxis].astype("uint8")

#Imprimir tamaño de los datos
print(x.shape)
print(y.shape)

#Normalizar datos
x[:, 0] = x[:, 0] - np.min(x[:, 0])
x[:, 0] = x[:, 0]/np.max(x[:, 0])
x[:, 1] = x[:, 1] - np.min(x[:, 1])
x[:, 1] = x[:, 1]/np.max(x[:, 1])

#Desordenar datos de entrada
arr = np.arange(x.shape[0])   #Crear vector de indices 
np.random.shuffle(arr)  #Ordenar vector de forma aleatoria

x = x[arr, :]
y = y[arr, :]

#print(y)

#Graficar informacion
plt.figure(figsize=(3,3))
plt.scatter(x[:, 0], x[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
plt.show()


#Separar datos en entrenamiento y validacion
x_train = x[0:270, :]
y_train = y[0:270, :]
x_val = x[270:, :]
y_val = y[270:, :]

print("Datos de entrenamiento: " + str(x_train.shape[0]))
print("Datos de valdiacion: " + str(x_val.shape[0]))


#Definir funciones auxiliares
def sigmoid(x):
    
    y = 1.0/(1 + np.exp(-x))
    
    return y

def softmax(x):
    
    y = np.exp(x)/np.sum(np.exp(x))
    
    return y

def cross_entropy(x):
    
    y = -np.log(x)
    
    return y



#Crear clase representativa de Perceptron multicapa (Clasificacion)
class perceptron_mult:
    
    #Crear constructor
    def __init__(self, d, ne, no, ns):
        
        #Parametros de la capa de entrada
        self.w1 = np.random.rand(ne, d) - 0.5
        self.b1 = np.random.rand(ne, 1) - 0.5
        
        #Parametros de la capa de oculta
        self.w2 = np.random.rand(no, ne) - 0.5
        self.b2 = np.random.rand(no, 1) - 0.5
        
        #Parametros de la capa de salida
        self.w3 = np.random.rand(ns, no) - 0.5
        self.b3 = np.random.rand(ns, 1) - 0.5
    
        #Inicializar atributos de error y rendimiento
        self.e = 0.0
        self.r = 0.0
        
    #Funcion forward (Paso hacia adelante)
    def forward(self, x):
        
        #Capa de entrada
        h1 = np.dot(self.w1, x) + self.b1 #(ne, 1)
        y1 = sigmoid(h1) #(ne, 1)
        
        #Capa oculta
        h2 = np.dot(self.w2, y1) + self.b2 #(no, 1)
        y2 = sigmoid(h2) #(no, 1)
        
        #Capa de salida
        h3 = np.dot(self.w3, y2) + self.b3 #(ns, 1)
        ym = softmax(h3)  #(ns, 1)
        
        return ym
    
    #Funcion de entrenamiento
    def train(self, x, y, x_val, y_val, Lr, epoch):
        
        #Inicializar medidas de error y rendimiento
        self.e = np.zeros(epoch)
        self.r = np.zeros(epoch)
        self.ev = np.zeros(epoch)
        self.rv = np.zeros(epoch)
        
        #Lazo de epocas
        for i in range(epoch):
            
            #Lazo de datos
            for j in range(x.shape[0]):
                
                #Leer entrada
                x_in = x[j, :]  #(d, )
                x_in = x_in[:, np.newaxis]  #(d, 1)
                
                #Capa de entrada
                h1 = np.dot(self.w1, x_in) + self.b1 #(ne, 1)
                y1 = sigmoid(h1) #(ne, 1)
        
                #Capa oculta
                h2 = np.dot(self.w2, y1) + self.b2 #(no, 1)
                y2 = sigmoid(h2) #(no, 1)

                #Capa de salida
                h3 = np.dot(self.w3, y2) + self.b3 #(ns, 1)
                ym = softmax(h3)  #(ns, 1)
                
                #Leer salida deseada
                c_corr = y[j]
                
                #Calcular error del modelo
                error = cross_entropy(ym[c_corr])
                
                #Derivadas de la funcion de error
                de_ym = - 1.0/ym[c_corr]  #(1)
                
                #Derivadas de la capa de salida
                
                #a) Funcion softmax
                dym_h3 = np.zeros(ym.shape)  #(ns, 1)
                for k in range(ym.shape[0]):
                    
                    #Posicion correcta
                    if (k == c_corr):
                        dym_h3[k, :] = ym[c_corr]*(1.0 - ym[c_corr])
                    else:
                        dym_h3[k, :] = -ym[c_corr]*ym[k, :]
                
                #b) Modelo parametrizado
                dh3_w3 = y2  #(no, 1)
                dh3_b3 = 1.0  #(1)
                dh3_y2 = self.w3  #(ns, no)
                
                #Derivadas de la capa oculta
                
                #a) Funcion sigmoide
                dy2_h2 = y2*(1.0 - y2)  #(no, 1)
                
                #b) Modelo parametrizado
                dh2_w2 = y1  #(ne, 1)
                dh2_b2 = 1.0 #(1)
                dh2_y1 = self.w2  #(no, ne)
                
                #Derivadas de la capa entrada
                
                #a) Funcion sigmoide
                dy1_h1 = y1*(1.0 - y1) #(ne, 1)
                
                #b) Modelo parametrizado
                dh1_w1 = x_in #(d, 1)
                dh1_b1 = 1.0  #(1.0)
                
                #Construir gradientes de capa de salida
                de_w3 = np.dot(de_ym*dym_h3, np.transpose(dh3_w3)) #(ns, no)
                        #(1)*(ns,1)*(no,1)
                        #(ns,1)*(1, no) = (ns, no)
                de_b3 = de_ym*dym_h3*dh3_b3 #(ns, 1)
                         #(1)*(ns, 1)*(1) 
                
                de_y2 = np.transpose(np.dot(np.transpose(de_ym*dym_h3), dh3_y2)) #(no, 1)
                        #(1)*(ns,1)*(ns,no)
                        #(1, ns)*(ns, no) = (1, no)'
                        
                #Construir gradientes de capa oculta
                de_w2 = np.dot(de_y2*dy2_h2, np.transpose(dh2_w2)) #(no, ne)
                        #(no,1)*(no,1)*(ne,1)
                        #(no,1)*(1,ne)
                
                de_b2 = de_y2*dy2_h2*dh2_b2 #(no, 1)
                        #(no,1)*(no,1)*(1)
                
                de_y1 = np.transpose(np.dot(np.transpose(de_y2*dy2_h2), dh2_y1)) #(ne, 1)
                        #(no,1)*(no,1)*(no, ne)
                        #(1,no)*(no,ne) = (1,ne)'
                        
                #Construir gradientes de capa de entrada
                de_w1 = np.dot(de_y1*dy1_h1, np.transpose(dh1_w1)) #(ne, d)
                        #(ne,1)*(ne,1)*(d,1)
                        #(ne,1)(1,d)
                de_b1 = de_y1*dy1_h1*dh1_b1 #(ne, 1)
                        #(ne,1)*(ne,1)*(1)
                        #(ne,1)*(1)
                        
                #Actualizar parametros de la red
                self.w3 = self.w3 - Lr*de_w3
                self.b3 = self.b3 - Lr*de_b3
                self.w2 = self.w2 - Lr*de_w2
                self.b2 = self.b2 - Lr*de_b2
                self.w1 = self.w1 - Lr*de_w1
                self.b1 = self.b1 - Lr*de_b1
                
                #Acumular error
                self.e[i] = self.e[i] + error
                
                #Verificar si modelo predijo la clase correcta
                if(np.argmax(ym) == c_corr):
                    
                    #Acumular acierto
                    self.r[i] = self.r[i] + 1
                
            #Calcular error y rendimiento de época
            self.e[i] = self.e[i]/x.shape[0]
            self.r[i] = self.r[i]/x.shape[0]
            
            #Evaluar modelo con datos de validacion
            for j in range(x_val.shape[0]):
                
                #Obtener entrada
                x_in = x_val[j, :]  #(d,)
                x_in = x_in[:, np.newaxis]  #(d, 1)
                
                #Someter modelo a datos
                prob = self.forward(x_in)
                
                #Obtener salida correcta
                c_corr = y_val[j]
                
                #Acumular error
                self.ev[i] = self.ev[i] + cross_entropy(prob[c_corr])
                
                #Verificar si modelo predijo la clase correcta
                if(np.argmax(prob) == c_corr):
                    
                    self.rv[i] = self.rv[i] + 1
                    
            #Calcular error y rendimiento de época
            self.ev[i] = self.ev[i]/x_val.shape[0]
            self.rv[i] = self.rv[i]/x_val.shape[0]



#Crear instancia de la red neuronal
redc = perceptron_mult(2, 10, 10, 3)

#Entrenar modelo
redc.train(x_train, y_train, x_val, y_val, 0.02, 1000)


#Crear variable para guardar salida del modelo
ym = np.zeros(x.shape[0])

#Evaluar modelo entrenado sobre datos de entrada
for i in range(x.shape[0]):
    
    #Extraer entrada de vector x
    x_in = x[i, :]  #(d, )
    x_in = x_in[:, np.newaxis]  #(d, 1)
    
    #Someter modelo a entrada
    prob = redc.forward(x_in)
    
    #Obtener prediccion del modelo
    ym[i] = np.argmax(prob)
    
#Graficar informacion
plt.figure(2, figsize=(3,3))
plt.scatter(x[:, 0], x[:, 1], c=ym, s=20, cmap=plt.cm.Spectral)
plt.show()

#Graficar curva de error
plt.figure(3, figsize=(3,3))
plt.plot(redc.e, 'b')
plt.plot(redc.ev, 'r')
plt.grid()
plt.show()

#Graficar curva de error
plt.figure(4, figsize=(3,3))
plt.plot(redc.r, 'b')
plt.plot(redc.rv, 'r')
plt.grid()
plt.show()


#Construir gradilla de prueba
x_gradilla= np.zeros([400, 2])

#Asignar valores a gradilla
for i in range(20):
    
    x_gradilla[i*20:(i+1)*20, 0] = i/20.0
    x_gradilla[i*20:(i+1)*20, 1] = np.linspace(0, 100, 20)/100.0

#Inicializar salidas del modelo
y_gradilla = np.zeros(x_gradilla.shape[0])
y_prob = np.zeros([x_gradilla.shape[0], 3])

#Someter modelo a gradilla de puntos
for i in range(x_gradilla.shape[0]):
    
    #Extraer entrada de vector x
    x_in = x_gradilla[i, :]  #(d, )
    x_in = x_in[:, np.newaxis]  #(d, 1)
    
    #Someter modelo a entrada
    y_prob[i, :] = np.transpose(redc.forward(x_in))
    
    #Obtener prediccion del modelo
    y_gradilla[i] = np.argmax(y_prob[i, :])
    
#Graficar datos
plt.figure(figsize=(5,5))
plt.scatter(x_gradilla[:, 0], x_gradilla[:, 1], c=y_gradilla, s=80, cmap=plt.cm.Spectral)
plt.show()


#Visualizar rectas (2 clusters)
fig = plt.figure()
bx = plt.axes(projection = '3d')
bx.scatter(x_gradilla[:, 0], x_gradilla[:, 1], y_prob[:, 0], marker='o', s=10, c='red')
bx.scatter(x_gradilla[:, 0], x_gradilla[:, 1], y_prob[:, 1], marker='o', s=10, c='blue')
bx.scatter(x_gradilla[:, 0], x_gradilla[:, 1], y_prob[:, 2], marker='o', s=10, c='yellow')
bx.scatter(x[:, 0], x[:, 1], np.zeros(x[:, 0].shape) + 0.5, marker='*', s=10, c='black')