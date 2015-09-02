import DPlib
from sklearn import neighbors, metrics, svm
client = 'AlienVault'
data_path = '/home/hugo/DATA/' 
data_file = 'BRENNT_' + client + '_Data.csv'
test_file = 'BRENNT_' + client + '_Test.csv'
aux_path = client + '/'
cat_list = [2,5,6,23,24,25,26,27]
stats_file = client + '.stats'
name_list = client + '.names'

DPlib.getLabels(data_path, data_file, cat_list, aux_path, stats_file)

DATA, LABEL = DPlib.getAllModData(data_path, data_file, aux_path, name_list, stats_file)
tDATA, tLABEL = DPlib.getAllModData(data_path, test_file, aux_path, name_list, stats_file)

clfkNNu = neighbors.KNeighborsClassifier(3, 'uniform', p=5)
clfkNNd = neighbors.KNeighborsClassifier(3, 'distance', p=5)
clfkNNc = neighbors.NearestCentroid()

clfkNNu.fit(DATA, LABEL)
clfkNNd.fit(DATA, LABEL)
clfkNNc.fit(DATA, LABEL)

pLABELkNNu = clfkNNu.predict(tDATA)
pLABELkNNd = clfkNNd.predict(tDATA)
pLABELkNNc = clfkNNc.predict(tDATA)

V = [pLABELkNNu, pLABELkNNd, pLABELkNNc]


pLABELmajority = []
for ii in range(len(V[0])):
    summ = 0
    for jj in range(3):
        summ = summ + V[jj][ii]
    pLABELmajority.append(int(summ/2.))

V.append(pLABELmajority)

fw = open(aux_path + client + '.predmaj','w')
fw.write(str(pLABELmajority).replace('[','').replace(']',''))
fw.close()
