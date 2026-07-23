#Krsinar Xavier - 26102387 

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import  KMeans
from sklearn.metrics import confusion_matrix


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#chargement du fichier et selection des donnees
etudiant = 26102387 
print(" Etudiant : KRSINAR Xavier, numero etudiant = ",etudiant)
np.random.seed(etudiant)


header = np.loadtxt("data.csv", max_rows=1, delimiter=",", dtype=str)
varbs = header[1:] #noms des genes
Xcomplet = np.loadtxt("data.csv", skiprows=1, delimiter=",", usecols=range(1, 20532)) #contient toutes les variables
nvars = 50 #nombre de variables selectiones, possibilite de changer


var = np.var(Xcomplet, axis=0) #variance
ind = np.argsort(var)[-nvars:]
#Variante pour selection AFD
# y_complet = np.loadtxt("labels.csv", delimiter=",", skiprows=1, dtype=str)[:, 1]
# score_afd, _ = f_classif(Xcomplet, y_complet) 
# score_afd = np.nan_to_num(score_afd) 
# ind = np.argsort(score_afd)[-nvars:]

#pour selection 200 premiere var 
# nvars = 200 
# ind = np.arange(nvars)
# X = Xcomplet[:, ind]
# varbs = varbs[ind]

X = Xcomplet[:, ind] #variables selectionnees
varbs = varbs[ind]
print("Sont lues les donnees correpondant aux nvar = ", nvars, "genes les plus variants et a la moitie des individus",
"(tiree aleatoirement sur la base du numero etudiant)")
nech=X.shape[0]//2
y =np.loadtxt("labels.csv",delimiter=",",skiprows=1,dtype=str)
per=np.random.permutation(X.shape[0])[:nech]
X,y = X[per,:], y[per,1]
print("Nombre de lignes, nombre de colonnes : ",X.shape)

# Pas besoin d'elimination des variables constantes, car forcement differentes

#--------------------------------------------
#-----------------ACP-----------------------
#--------------------------------------------

print("------ACP------")

def stdise(X):
  """Routine de standardisation
     On pourrrait utiliser scikitlearn: Exemple:
       from sklearn.preprocessing import StandardScaler
       X=np.arange(12).reshape((4,3))
       print(StandardScaler().fit_transform(X))
  """
  Xs=X.astype(float)
  mk=np.mean(Xs,axis=0)
  sk=np.maximum(np.std(Xs,axis=0),10*np.finfo(float).eps)
  Xs=(X-mk)/sk
  return Xs, mk, sk
# SVD. Axes Composantes

Xs, mk,sk=stdise(X)

(U,D,VT) = np.linalg.svd(Xs,full_matrices=False)
V=VT.T
# Premieres composantes principales
C1 = D[0]*U[:,0]
C2 = D[1]*U[:,1]
C3 = D[2]*U[:,2]
# Axes principaux modifies pour le cercle des correlations
A1 = D[0]*V[:,0]/np.sqrt(np.shape(X)[0])
A2 = D[1]*V[:,1]/np.sqrt(np.shape(X)[0])
# Graphiques
plt.close('all')
plt.figure()
plt.title('Representation des individus dans le plan (C1,C2)')
if y is None:
  plt.scatter(C1,C2)
else:
  vlab=np.unique(y)
  lv=len(vlab)
  for i,vl in enumerate(vlab):
    l=y==vl
    plt.scatter(C1[l],C2[l],s=47,label=vl)#,color=cols[i])
  plt.legend(title="Type de cancer")
plt.xlabel('C1')
plt.ylabel('C2')

# Inerties
plt.figure()
plt.bar(np.arange(np.shape(D)[0])+1,100*D**2/sum(D**2))
plt.title('Inerties en %')

# Cercle des correlations
if not varbs is None:
  plt.figure()
  plt.title('Cercle des correlations')
  Z = np.linspace(-np.pi, np.pi, 256,endpoint=True)
  C,S = np.cos(Z), np.sin(Z)
  plt.plot(C,S,c='black',lw=.7)
  plt.axvline(c='black',ls='dashed',lw=1)
  plt.axhline(c='black',ls='dashed',lw=1)
  for i, txt in enumerate(varbs):
    plt.arrow(0,0,A1[i],A2[i], length_includes_head=True,
            head_width=0.025, head_length=.05)
    plt.annotate(txt, (A1[i]+.01,A2[i]+.01),fontsize=12)
  plt.xlabel('C1')
  plt.ylabel('C2')

#------------------------------------------------
#----------------clustering-----------------------
#------------------------------------------------

# On utilise les donnees standardisees de l'ACP (Xs)
# ainsi que les labels (y)

print("\n******* Classification Ascendante hiérarchique ******* \n")

# Calcul de l'arbre 
M=linkage(Xs,method='ward',metric='euclidean')

# Tracé de l'arbre 
seuil=40 # bien au vu de notre dentogramme
plt.figure()
plt.title('CAH. Visualisation des classes au seuil de '+str(seuil))
# orientation vers le haut (top) pour plus de lisibilite
d=dendrogram(M,labels=None,no_labels=True,orientation='top',color_threshold=seuil)
plt.show()

##### Récupération des groupes 
groupes=fcluster(M,t=seuil,criterion='distance')
# nombre de groupes
print('Nombre de groupes trouvés par CAH :', np.max(groupes))

#### Décroissance des variances intraclasse 
VI=np.cumsum(M[:,2]**2)/2
plt.figure()
plt.plot(np.arange(len(VI))+1,np.flip(VI,axis=0))
plt.xlabel("Nombre de classes")
plt.ylabel("Variance intraclasse")
plt.xlim(1, 15)
plt.grid(True)
plt.show()


#k mean
print("\n******* Kmeans  ******* \n")


nclus=5 #car 5 cancer
k_means = KMeans(init='k-means++', n_clusters=nclus, n_init=10)
k_means.fit(Xs)
cl = k_means.labels_ # classes "prédites"
for k in range(nclus):
    print('Classe '+str(k+1).ljust(3,' ')+': ', end='')
    print(*y[np.where(cl==k)])

print("\n******* Comparaison des inerties ******* \n")

print("Inertie Kmeans",nclus,"centres: ",k_means.inertia_)
print("Inertie CAH",nclus,"classes: ",VI[-nclus])

# --- analyse et matrice

# Conversion des labels texte (y) en int 
noms_cancers, labels = np.unique(y, return_inverse=True)

# Calcul de l'étiquette majoritaire de chaque classe et du taux d'erreur 
# On fabrique le tableau maj_lab qui a un nurmero de classe 
# renvoie l'etiquette correspondante.
maj_lab=np.arange(k_means.n_clusters) # Initialisation de tableau


for k in range(k_means.n_clusters):
  counts=np.unique(labels[cl==k],return_counts=True) # Nb d'occurences de chaque label
  imax=np.argmax(counts[1]) # Recherche du majoritaire dans k
  maj_lab[k]=counts[0][imax] # Son étiquette ya majoritaire dans k
print('\n')
print("Classe".ljust(23,'.')+" ",end='')
print(*range(k_means.n_clusters),end='')
print("\n"+"Etiquette majoritaire".ljust(23,'.')+" ",end='')
print(*(maj_lab))
err=sum(labels!=maj_lab[cl])/len(cl)
print("Taux de mal classés:",err.round(3))

conf_mat =  confusion_matrix(labels,maj_lab[cl])
plt.rcParams.update({'figure.figsize': (5, 5), 'font.size': 10})
ConfusionMatrixDisplay(conf_mat,display_labels=noms_cancers).plot(cmap='Blues')
im = plt.gca().images[-1]#.colorbar.remove()
plt.rcdefaults()
plt.show()

# Représentation en barplot 
def BarPlotMat(M):
# Fait un barplot pour chaque colonne de M.
# La couleur correspond à l'indice, la hauteur à la valeur
  I=M.shape[0]
  J=M.shape[1]
  ind = np.arange(J)
  haut = 0*M[0,:]
  for i in range(I):
    plt.bar(ind,M[i,:],bottom=haut,color=plt.cm.inferno(i/(I-1)))
    haut += M[i,:]

fig=plt.figure(3)
plt.clf()
conf_mat =  confusion_matrix(labels,cl)
# Nettoyage des lignes et colonnes vides 
conf_mat=conf_mat[np.sum(conf_mat,axis=1)>0,:]
conf_mat=conf_mat[:,np.sum(conf_mat,axis=0)>0]
BarPlotMat(conf_mat)
plt.xlabel('Classe')
plt.ylabel('Répartition des étiquettes')
plt.title('Répartition dans chaque classe')
plt.legend(noms_cancers) 
plt.show()

print("Matrice associée (conf_mat[etiq, classe]):\n")
print(conf_mat)
print("   Une ligne = un digit (ici un cancer)\n   Une colonne = une classe\n")

#---------------------------------------------------------------------------
#---------------------------analyse discriminante----------------------------
#---------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")
print("\n--- Analyse discriminante linéaire ---")

lda = LinearDiscriminantAnalysis(tol=1e-5)
lda.fit(Xs, y)
yhat = lda.predict(Xs)
errl = sum(y != yhat) / len(y)
print("Taux d'erreur: ", round(errl, 3))





# Affichage graphique 
plt.rcParams.update({'figure.figsize': (5, 5), 'font.size': 10})
conf_mat = confusion_matrix(y, yhat)
disp = ConfusionMatrixDisplay(conf_mat, display_labels=lda.classes_)
disp.plot(cmap='YlOrBr', colorbar=False, xticks_rotation='vertical', ax=plt.gca())
plt.title("Matrice de confusion LDA (sans validation croisee)")
plt.rcdefaults()
plt.show()





# # Pour tracer soi-même la matrice de confusion en format texte 
# print("Matrice de confusion (sans validation croisée)\n")
# conf = conf_mat.astype(str)
# headers_col = np.array([["yh=" + c for c in lda.classes_]])
# headers_row = np.array(["y=" + c for c in lda.classes_])

# #Montage de la matrice texte 
# conf = np.c_[headers_row, conf]
# top_row = np.r_[['   '], headers_col[0]] 
# conf = np.r_[ [top_row], conf ]

# conft = conf.T
# for i, b in enumerate(conft):
#     l = max(len(i) for i in b) 
#     conft[i] = np.char.ljust(b, l + 1, ' ')
# conf = conft.T
# for row in conf: print(*row)




# sans validation croisee
C = lda.fit_transform(Xs, y) 
C1 = C[:, 0] 
tmp = [C1[y == cls] for cls in lda.classes_]

plt.figure()
plt.boxplot(tmp, labels=lda.classes_, widths=.8)
plt.title("Distribution de la variable discriminante (Axe 1)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

C2 = C[:, 1] 
tmp = [C2[y == cls] for cls in lda.classes_]

plt.figure()
plt.boxplot(tmp, labels=lda.classes_, widths=.8)
plt.title("Distribution de la variable discriminante (Axe 2)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

C3 = C[:, 2] 
tmp = [C3[y == cls] for cls in lda.classes_]

plt.figure()
plt.boxplot(tmp, labels=lda.classes_, widths=.8)
plt.title("Distribution de la variable discriminante (Axe 3)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

C4 = C[:, 3] 
tmp = [C4[y == cls] for cls in lda.classes_]

plt.figure()
plt.boxplot(tmp, labels=lda.classes_, widths=.8)
plt.title("Distribution de la variable discriminante (Axe 4)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# # Avec validation croisée 
# ntest = np.floor(len(y) / 2).astype(int)
# per = np.random.permutation(len(y))
# lt, la = per[:ntest], per[ntest:]
# Xa, Xt = X[la, :], X[lt, :]
# ya, yt = y[la], y[lt]

# lda.fit(Xa, ya)
# yhat = lda.predict(Xt) # yhat recalculé sur le test
# errl = sum(yt != yhat) / len(yt)
# print("Taux d'erreur (Validation simple 50%): ", round(errl, 3))


# Analyse discriminante quadratique -------------------------------------------
print("\n--- Analyse discriminante quadratique ---")

qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(Xs, y)
yhat = qda.predict(Xs)
errq = sum(y != yhat) / len(y)
print("Taux d'erreur: ", round(errq, 3))

# Affichage graphique QDA (non croise)
plt.rcParams.update({'figure.figsize': (5, 5), 'font.size': 10})
conf_mat_q = confusion_matrix(y, yhat)
disp_q = ConfusionMatrixDisplay(conf_mat_q, display_labels=qda.classes_)
disp_q.plot(cmap='YlOrBr', colorbar=False, xticks_rotation='vertical', ax=plt.gca())
plt.title("Matrice de confusion QDA (sans validation croisee")
plt.rcdefaults()
plt.show()




#fct validation croisee
def validation_croisee(Modele, X_selectionne, y, ntest, n_iter):
    classes = np.unique(y)
    mat_cumul = np.zeros((len(classes), len(classes)), dtype=int)
    
    for i in range(n_iter):
        per = np.random.permutation(len(y))
        lt, la = per[:ntest], per[ntest:]
        Xa_raw, Xt_raw = X_selectionne[la, :], X_selectionne[lt, :]
        ya, yt = y[la], y[lt]
        mk = np.mean(Xa_raw, axis=0)
        sk = np.maximum(np.std(Xa_raw, axis=0), 10 * np.finfo(float).eps)
        Xa_std = (Xa_raw - mk) / sk
        Xt_std = (Xt_raw - mk) / sk
        Modele.fit(Xa_std, ya)
        yhat = Modele.predict(Xt_std)
        
        mat_cumul += confusion_matrix(yt, yhat, labels=classes)
        
    return mat_cumul

print("\n--- Validation croisee - 50 iterations ---")

ntest = np.floor(len(y)/2).astype(int)
n_iter = 50
mat_lda = validation_croisee(LinearDiscriminantAnalysis(tol=1e-5), Xs, y, ntest, n_iter)
err_lda = (mat_lda.sum() - np.trace(mat_lda)) / mat_lda.sum()
print("Taux d'erreur MOYEN LDA: ", round(err_lda, 3))




# --- QDA ---
mat_qda = validation_croisee(QuadraticDiscriminantAnalysis(reg_param=0.1), Xs, y, ntest, n_iter)

err_qda = (mat_qda.sum() - np.trace(mat_qda)) / mat_qda.sum()
print("Taux d'erreur MOYEN QDA: ", round(err_qda, 3))


#affichage matrice confusion avec validation croisee
# LDA
plt.rcParams.update({'figure.figsize': (5, 5), 'font.size': 10})
disp_ldav = ConfusionMatrixDisplay(mat_lda, display_labels=lda.classes_)
disp_ldav.plot(cmap='YlOrBr', colorbar=False, xticks_rotation='vertical', ax=plt.gca())
plt.title("Matrice de confusion LDA (Validation Croisée 50 iter)")
plt.rcdefaults()
plt.show()


#QDA
plt.rcParams.update({'figure.figsize': (5, 5), 'font.size': 10})
disp_qdav = ConfusionMatrixDisplay(mat_qda, display_labels=qda.classes_)
disp_qdav.plot(cmap='YlOrBr', colorbar=False, xticks_rotation='vertical', ax=plt.gca())
plt.title("Matrice de confusion QDA (Validation Croisée 50 iter)")
plt.rcdefaults()
plt.show()