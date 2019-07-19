import scipy.io.wavfile as wav
import scipy.signal as filter
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as py
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import delta

from scipy.cluster.vq import vq, kmeans, whiten

rate, signal = wav.read('banana.wav')
mfcc_feat = mfcc(signal, rate)
d_mfcc_feat = delta(mfcc_feat, 2)
dd_mfcc_feat = delta(d_mfcc_feat, 2)
mfcc_39 = py.hstack([mfcc_feat, d_mfcc_feat, dd_mfcc_feat])

sdarr = mfcc_39.T.reshape(-1, 1)
plt.plot(sdarr)
plt.show()
print(sdarr.shape)

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(sdarr)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 10), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()

n_clusters = 3
print(n_clusters)

py.random.seed(0)

k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
k_means.fit(sdarr)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_


print('value: ',values, "ukuran", values.shape)
py.savetxt('value.npy', values)
print('label: ', labels, "ukuran", labels.shape)
py.savetxt('labels.npy', labels)

phone = ['w', 'ah', 'n']
phoneme_dict = dict(zip(values,phone ))


print(phoneme_dict)


#a = py.load('model/codebook.npy')
#print(a)

#df = pd.DataFrame(data=sdarr)
#print(df)


#print(mfcc_39_feat.dtype)
#print(mfcc_39_feat.shape)
#print("rows : ", mfcc_39_feat.shape[0])
#print("cols : ", mfcc_39_feat.shape[1])
#print("length" ,len(mfcc_39_feat))

#print(mfcc_39_feat)

#plt.plot(mfcc_feat)
#plt.imshow(mfcc_39_feat, origin='lower')
#ax = plt.gca()
#ax.invert_yaxis()
#plt.show()

#col1 = mfcc_39_feat[:,0]
#print(col1.shape)
#plt.plot(col1)

#plt.plot(mfcc_39_feat[:,1])
#plt.show()

#plt.scatter(x=mfcc_39_feat[:0], y=mfcc_39_feat[:0], cmap='rainbow')
#plt.show()

'''
ax1 = plt.subplot2grid((10, 6), (0, 0), rowspan=2, colspan=2)
plt.title("signal asli")
ax2 = plt.subplot2grid((10, 6), (3, 0), rowspan=2, colspan=2)
plt.title("signal MFCC")
ax3 = plt.subplot2grid((10, 6), (6, 0), rowspan=2, colspan=2)
plt.title("signal delta MFCC")
ax4 = plt.subplot2grid((10, 6), (0, 3), rowspan=2, colspan=2)
plt.title("signal delta + delta MFCC")
ax5 = plt.subplot2grid((10, 6), (3, 3), rowspan=2, colspan=2)
plt.title("signal MFCC 39 Feature")

ax1.plot(signal)
ax2.imshow(mfcc_feat, cmap='hot', interpolation='nearest')
ax3.imshow(d_mfcc_feat, cmap='hot', interpolation='nearest')
ax4.imshow(dd_mfcc_feat, cmap='hot', interpolation='nearest')
ax5.imshow(mfcc_39_feat,cmap='hot', interpolation='nearest')

plt.show()

plt.scatter(x = mfcc_39_feat[0], y = mfcc_39_feat[1])
plt.show()'''
