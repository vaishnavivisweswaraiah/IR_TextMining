from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings

'TF-IDF training data file'
training_data_file_TFIDF = "training_data_file.TFIDF"
feature_vectors_TFIDF, targets_TFIDF = load_svmlight_file(training_data_file_TFIDF)
train_feature=feature_vectors_TFIDF
train_target=targets_TFIDF


X = train_feature
classification_labels = train_target
X_new1 = SelectKBest(chi2, k=100).fit_transform(X, classification_labels)
n_cluster=[]
KSilhouette=[]
HSilhouette=[]
Knormalized_mutual_info_score=[]
Hnormalized_mutual_info_score=[]
print("Clustering and Quality evaluation is in Progress for feature count k=100 for clusters count in range[2,25]")
for k in range(2,25):
    warnings.filterwarnings("ignore")
    n_cluster.append(k)
    'K means clustering '
    kmeans_model = KMeans(n_clusters=k).fit(X_new1)
    Kclustering_labels = kmeans_model.labels_
    '''This function returns the Silhouette Coefficient for each sample.   
    The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.'''
    Silhouette=metrics.silhouette_score(X_new1, Kclustering_labels, metric='euclidean')
    KSilhouette.append(Silhouette)
    '''normalized_mutual_info_score score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling'''
    normalized_mutual_info_score=metrics.normalized_mutual_info_score(classification_labels, Kclustering_labels,average_method='arithmetic')
    Knormalized_mutual_info_score.append(normalized_mutual_info_score)
    'Heirarical bottom up clustering'
    single_linkage_model = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X_new1.toarray())
    Hclustering_labels = single_linkage_model.labels_
    '''This function returns the Silhouette Coefficient for each sample.   
    The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.'''
    Silhouette=metrics.silhouette_score(X_new1, Hclustering_labels, metric='euclidean')
    HSilhouette.append(Silhouette)
    '''normalized_mutual_info_score score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling'''
    normalized_mutual_info_score=metrics.normalized_mutual_info_score(classification_labels, Hclustering_labels,average_method='arithmetic')
    Hnormalized_mutual_info_score.append(normalized_mutual_info_score)
   
print("Cluster range",n_cluster)
print("Silhouette scores for K-means clustering",KSilhouette)
print("Silhouette scores for AgglomerativeClustering(hierarchical clustering)",HSilhouette)
print("normalized_mutual_info_score for K-means clustering",Knormalized_mutual_info_score)
print("normalized_mutual_info_score for AgglomerativeClustering(hierarchical clustering)",Hnormalized_mutual_info_score)
plt.figure()
plt.subplot(221)
plt.plot(n_cluster,KSilhouette,'o-', color="r",label="KMeansClustering")
plt.plot(n_cluster,HSilhouette,'.-', color="g",label="HeirarichalClustering")
plt.title("Silhouette Measure")
plt.xlabel("K-values(number of clusters)")   
plt.axis([1,25,0,1.1])
plt.legend(loc="best",fontsize = 'x-small')
plt.ylabel("measures")
'second plot for normalised mutual info score'
plt.subplot(222)
plt.plot(n_cluster,Knormalized_mutual_info_score,'o-', color="r",label="KMeansClustering")
plt.plot(n_cluster,Hnormalized_mutual_info_score,'.-', color="g",label="HeirarichalClustering")
plt.title("normalized_mutual_info_score measure")
plt.xlabel("K-values(number of clusters)")   
plt.axis([1,25,0,0.20])
plt.legend(loc="best",fontsize = 'x-small')
'y labels for each plots'
plt.ylabel("measures")
#plt.grid(True)
plt.show()
