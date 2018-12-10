import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np





#leo la imagen 
I = cv.imread("C:/Users/Victor Manuel/Documents/John/Imagen_pruebas.jpg")

#Convierto de BGR a RGB
img = cv.cvtColor(I, cv.COLOR_BGR2RGB) 

#CONOCEMOS RESOLUCION DE LA IMAGEN
print (I.shape)

#TOMAMOS UN UNICO CANAL
R_I = img[:,:,0]
print (R_I.shape) 

#VISUALIZAMOS LA IMAGEN
plt.imshow(R_I,cmap='gray')
plt.show();
#NUMERO DE FILAS Y COLUMNAS
[n, m] = np.shape(R_I)
print ([n,m])

#SE VISUALIZAN LOS PUNTOS DE INTERES EN EL HISTOGRAMA (QUE TIENEN VALOR CERO)
plt.hist(R_I.ravel(),256,[0,256]); plt.show()

#IDENTIFICA LAS COORDENADAS DE LOS PUNTOS DONDE 
indices = np.where(R_I == [0])
#print (indices[0]) #columnas
#print (indices[1]) #filas

#GRAFICAMOS LAS COORDENADAS DE LOS PUNTOS DE INTERES
fig = plt.figure()
plt.scatter(indices[1], indices[0]) #1 filas , o columnas
plt.title('Datos', fontsize = 14)
plt.show();

#CONCATENAMOS FILAS CON COLUMNAS
cordenadas=np.column_stack((indices[1], indices[0]))
#print(cordenadas)

#Dimensión de los datos de interes
[nf, nc] = np.shape(cordenadas)
print (nf,nc)

#transformamos los datos restandole a las coordenadas el promedio
#hallamos el promedio
prom=cordenadas.mean(axis=0) 
prom=np.rint(prom) 

#print(prom)
#Aproximamos al entero mas cercano
#np.around(prom)
#A los datos les restamos el promedio
Datos_trans=cordenadas-np.around(prom)

#Datos_trans=cordenadas-prom
#print (Datos_trans)

#GRAFICAMOS LAS COORDENADAS DE LOS PUNTOS DE INTERES CENTRADOS
fig = plt.figure()
plt.scatter(Datos_trans[:,0], Datos_trans[:,1]) 
plt.title('Datos Centrados', fontsize = 14)
plt.show();

#calculamos la matriz de covarianza
cov_mat = np.cov(Datos_trans.T)
print(cov_mat)

#Se determinan los eigen valores y eigen vectores
[eigen_val, eigen_vec] = np.linalg.eigh(cov_mat)

print(eigen_val)

#print(eigen_val[0])

print(eigen_vec)


#grafica de la funciòn
def f1(x):
 return (eigen_vec[0,0]/eigen_vec[0,1])*x
def f2(x):
   return (eigen_vec[1,0]/eigen_vec[1,1])*x
# Valores del eje X que toma el gráfico.
pendiente=eigen_vec[1,0]/eigen_vec[1,1]
print (pendiente)
#GRAFICAMOS LAS COORDENADAS DE LOS PUNTOS DE INTERES CENTRADOS
#fig = plt.figure()


plt.scatter(Datos_trans[:,0], Datos_trans[:,1]) 
plt.title('Datos Centrados', fontsize = 14)
x=np.linspace(-160,160,321)
plt.plot(x, [f1(i) for i in x])
plt.plot(x, [f2(i) for i in x])
plt.show();



   