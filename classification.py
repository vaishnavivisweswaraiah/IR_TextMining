
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
'Term frequency training)data_file'
training_data_file_tf = "training_data_file.TF"
feature_vectors_tf, targets_tf = load_svmlight_file(training_data_file_tf)
'''splitting the training data and test data '''
trainsize_tf=round(len(targets_tf)*1.0)
train_feature_tf=feature_vectors_tf[:trainsize_tf]
train_target_tf=targets_tf[:trainsize_tf]
test_feature_tf=feature_vectors_tf[trainsize_tf:]
test_target_tf=targets_tf[trainsize_tf:]
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

#print(train_target_TFIDF.shape,train_feature_TFIDF.shape)
#print(train_target_IDF.shape)

classifers={"Multinomialclassifier":MultinomialNB(),"Bernoulliclassifier":BernoulliNB(alpha=0.001),"KNeighborsClassifier":KNeighborsClassifier(weights='distance'),
            "SVCclassifer":SVC(gamma='scale',decision_function_shape='ovo')}

scoring = ['f1_macro','precision_macro', 'recall_macro']

def ClassificationModel():

    for name,classifer in classifers.items():
            print("***********"+name+"***********")
            crossval=classifer
            warnings.filterwarnings("ignore")
            if name=='Multinomialclassifier':
                print(name)
                train_feature=train_feature_tf
                train_target=train_target_tf
            if name=='Bernoulliclassifier':
                print(name)
                train_feature=train_feature_IDF
                train_target=train_target_IDF
            if (name=='KNeighborsClassifier'):
                print(name)
                train_feature=train_feature_TFIDF
                train_target=train_target_TFIDF
            if name=='SVCclassifer':
                print(name)
                train_feature=train_feature_TFIDF
                train_target=train_target_TFIDF
            scores = cross_validate(crossval, train_feature, train_target, scoring=scoring,cv=5, return_train_score=False)
            #print(scores.keys()) 
            print(" F1_macro Accuracy: %0.2f (+/- %0.2f)" % (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2))
            print(" precision_macro Accuracy: %0.2f (+/- %0.2f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2))
            print(" recall_macro Accuracy: %0.2f (+/- %0.2f)" % (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2))
            #print(classifer,scores['test_f1_macro'])

def test():
    '''Evaluating the system on test set and see mean square error cost function'''
    '''splitting the training data and test data 80 :20'''
    trainsize_tf=int(len(targets_tf)*0.8)
    train_feature_tf=feature_vectors_tf[:trainsize_tf]
    train_target_tf=targets_tf[:trainsize_tf]
    test_feature_tf=feature_vectors_tf[trainsize_tf:]
    test_target_tf=targets_tf[trainsize_tf:]
    'IDF training data file'
    training_data_file_IDF = "training_data_file.IDF"
    feature_vectors_IDF, targets_IDF = load_svmlight_file(training_data_file_IDF)
    trainsize_IDF=int(len(targets_IDF)*0.8)
    train_feature_IDF=feature_vectors_IDF[:trainsize_IDF]
    train_target_IDF=targets_IDF[:trainsize_IDF]
    test_feature_IDF=feature_vectors_IDF[trainsize_IDF:]
    test_target_IDF=targets_IDF[trainsize_IDF:]
    
    'TF-IDF training data file'
    training_data_file_TFIDF = "training_data_file.TFIDF"
    feature_vectors_TFIDF, targets_TFIDF = load_svmlight_file(training_data_file_TFIDF)
    trainsize_TFIDF=int(len(targets_TFIDF)*0.8)
    train_feature_TFIDF=feature_vectors_TFIDF[:trainsize_TFIDF]
    train_target_TFIDF=targets_TFIDF[:trainsize_TFIDF]
    test_feature_TFIDF=feature_vectors_TFIDF[trainsize_TFIDF:]
    test_target_TFIDF=targets_TFIDF[trainsize_TFIDF:]
    #print(test_feature_TFIDF.shape,test_target_TFIDF.shape)
    for name,classifer in classifers.items():
        clf=classifer
        warnings.filterwarnings("ignore")
        if name=='Multinomialclassifier':
            print(name+"Prediction with TF features ")
            train_feature=train_feature_tf
            train_target=train_target_tf
            test_feature=test_feature_tf
            test_target=test_target_tf
        if name=='Bernoulliclassifier':
            print(name+"Prediction with IDF features ")
            train_feature=train_feature_IDF
            train_target=train_target_IDF
            test_feature=test_feature_IDF
            test_target=test_target_IDF
        if (name=='KNeighborsClassifier'):
            print(name+"Prediction with TFIDF features ")
            train_feature=train_feature_TFIDF
            train_target=train_target_TFIDF
            test_feature=test_feature_TFIDF
            test_target=test_target_TFIDF
            #print(test_feature.shape,test_target.shape)
        if name=='SVCclassifer':
            print(name+"Prediction with TFIDF features ")
            train_feature=train_feature_TFIDF
            train_target=train_target_TFIDF
            test_feature=test_feature_TFIDF
            test_target=test_target_TFIDF

    
        clf.fit(train_feature,train_target)
        Y_predit=clf.predict(test_feature[315])
        test_final_Prediction=clf.predict(test_feature)
        'calculating square root of mean squared error training on train data test on test set'
        test_final_rmse=np.sqrt(mean_squared_error(test_target, test_final_Prediction))
        print("Test data Predicted value",Y_predit,"Expected value",test_target[315])
        Y_predit=clf.predict(train_feature[0])
        train_final_Prediction=clf.predict(train_feature)
        'calculating square root of mean squared error training and evalutation on training data'
        train_final_rmse=np.sqrt(mean_squared_error(train_target, train_final_Prediction))
        print("Training Data Predicted value",Y_predit,"Expected value",train_target[0])
        print("RMSE on training and test data",train_final_rmse,test_final_rmse)
if __name__ == '__main__':
    #test()
    ClassificationModel()
