from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings

'Term frequency training)data_file'
training_data_file_tf = "training_data_file.TF"
feature_vectors_tf, targets_tf = load_svmlight_file(training_data_file_tf)

train_feature_tf=feature_vectors_tf
train_target_tf=targets_tf

'IDF training data file'
training_data_file_IDF = "training_data_file.IDF"
feature_vectors_IDF, targets_IDF = load_svmlight_file(training_data_file_IDF)

train_feature_IDF=feature_vectors_IDF
train_target_IDF=targets_IDF

'TF-IDF training data file'
training_data_file_TFIDF = "training_data_file.TFIDF"
feature_vectors_TFIDF, targets_TFIDF = load_svmlight_file(training_data_file_TFIDF)

train_feature_TFIDF=feature_vectors_TFIDF
train_target_TFIDF=targets_TFIDF

clf = MultinomialNB()
clf1 = BernoulliNB(alpha=0.01)
clf2 = KNeighborsClassifier()
clf3 = SVC(gamma='scale',decision_function_shape='ovo')

#X = train_feature
#y = train_target
k1=[100,5000,30000, 32643] 
K_xaxis=k1

def Featureselection_F1Score(Featureselection,title,subplot):
    M_F1Score=[]
    B_F1Score=[]
    K_F1Score=[]
    SVC_F1Score=[]
    plt.figure(1)
    for k in k1:
        warnings.filterwarnings("ignore")
        if Featureselection=='CHI':
            X_Chi = SelectKBest(chi2, k=k).fit_transform(train_feature_tf, train_target_tf)
            'multinomial classifer for chi square feature selection method'
            scores = cross_val_score(clf, X_Chi, train_target_tf, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            M_F1Score.append(F1_scoreMean)
            X_Chi = SelectKBest(chi2, k=k).fit_transform(train_feature_IDF, train_target_IDF)
            'Bernoulli classifer for chi square feature selection method'
            scores = cross_val_score(clf1, X_Chi, train_target_IDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            B_F1Score.append(F1_scoreMean)
            X_Chi = SelectKBest(chi2, k=k).fit_transform(train_feature_TFIDF, train_target_TFIDF)
            'Kneighbours for chi square feature selection method'
            scores = cross_val_score(clf2, X_Chi, train_target_TFIDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            K_F1Score.append(F1_scoreMean)
            X_Chi = SelectKBest(chi2, k=k).fit_transform(train_feature_TFIDF, train_target_TFIDF)
            'SVC for chi square feature selection method'
            scores = cross_val_score(clf3, X_Chi, train_target_TFIDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            SVC_F1Score.append(F1_scoreMean)
        if Featureselection=='MUL':
            X_Mut = SelectKBest(mutual_info_classif, k=k).fit_transform(train_feature_tf, train_target_tf)
            'multinomial classifer for mutual information feature selection method'
            scores = cross_val_score(clf, X_Mut, train_target_tf, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            M_F1Score.append(F1_scoreMean)
            X_Mut = SelectKBest(mutual_info_classif, k=k).fit_transform(train_feature_IDF, train_target_IDF)
            'Bernoulli classifer for mutual information feature selection method'
            scores = cross_val_score(clf1, X_Mut, train_target_IDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            B_F1Score.append(F1_scoreMean)
            X_Mut = SelectKBest(mutual_info_classif, k=k).fit_transform(train_feature_TFIDF, train_target_TFIDF)
            'Kneighbours for mutual information feature selection method'
            scores = cross_val_score(clf2, X_Mut, train_target_TFIDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            K_F1Score.append(F1_scoreMean)
            X_Mut = SelectKBest(mutual_info_classif, k=k).fit_transform(train_feature_TFIDF, train_target_TFIDF)
            'SVC for mutual information feature selection method'
            scores = cross_val_score(clf3, X_Mut, train_target_TFIDF, cv=5, scoring='f1_macro')
            F1_scoreMean=scores.mean()
            SVC_F1Score.append(F1_scoreMean)  
    print("K values",K_xaxis,"\n","MultinomialNB",M_F1Score,"\n","BernoulliNB",B_F1Score,"\n","KNeighborsClassifier",K_F1Score,"\n","SVM",SVC_F1Score)
    'plotting graph for different values of k and average f1-macro score for Chisquared method and mutual information method'
    plt.subplot(subplot) 
    plt.plot(K_xaxis,M_F1Score,'.-', color="r",label="MultinomialNB")
    plt.plot(K_xaxis,B_F1Score,'.-', color="g",label="BernoulliNB")
    plt.plot(K_xaxis,K_F1Score,'.-', color="b",label="KNeighborsClassifier")
    plt.plot(K_xaxis,SVC_F1Score,'o-', color="g",label="SVC")
    plt.title(title)
    plt.xlabel("K-values(features)")   
    plt.axis([0,38000,0,1])
    plt.legend(loc="best",fontsize = 'xx-small')
    plt.ylabel("Average F1_macro")
    #plt.grid(True)
    return plt
def Ftest():
     X_Chi = SelectKBest(chi2, k=4).fit_transform(train_feature_tf, train_target_tf)
     X_Mut = SelectKBest(mutual_info_classif, k=10).fit_transform(train_feature_tf, train_target_tf)
     print(X_Chi)
     print("*******")
     print(X_Mut)
if __name__ == '__main__':
    print("chi-square feature selection for different k values is in progress")
    Featureselection_F1Score('CHI','chisquared method',int(221))
    print("mutual Information feature selection for different k values is in progress")
    Featureselection_F1Score('MUL','mutual information selection',int(222))
    plt.show()
    
    #Ftest()
