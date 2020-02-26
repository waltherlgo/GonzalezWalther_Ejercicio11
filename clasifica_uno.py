import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
target=target==1
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]
Data_train=x_train@vectores.T
Data_test=x_test@vectores.T

clf = LinearDiscriminantAnalysis()
SC=np.zeros(38)
SCTe=np.zeros(38)
NSC=np.zeros(38)
NSCTe=np.zeros(38)
for i in range(3,41):
    clf.fit(Data_train[:,:i+1], y_train)
    y_pred=clf.predict(Data_train[:,:i+1])
    yt_pred=clf.predict(Data_test[:,:i+1])
    SCTe[i-3]=f1_score(y_test,yt_pred)
    SC[i-3]=f1_score(y_train,y_pred)
    NSCTe[i-3]=f1_score(y_test,yt_pred,pos_label=False)
    NSC[i-3]=f1_score(y_train,y_pred,pos_label=False)

plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
plt.scatter(range(3,41),SC)
plt.scatter(range(3,41),SCTe)
plt.legend(["Train","Test"])
plt.title("Clasificación Uno")
plt.xlabel("Numero de Componentes PCA")
plt.ylabel("F1 score")
plt.subplot(1,2,2)
plt.scatter(range(3,41),NSC)
plt.scatter(range(3,41),NSCTe)
plt.legend(["Train","Test"])
plt.title("Clasificación Otros")
plt.xlabel("Numero de Componentes PCA")
plt.ylabel("F1 score")
plt.savefig("F1_score_LinearDiscriminantAnalysis.png")
plt.show()