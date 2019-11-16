'''

Index structure:

    The Index class contains a list of IndexItems, stored in a dictionary type for easier access

    each IndexItem contains the term and a set of PostingItems

    each PostingItem contains a document ID and a list of positions that the term occurs

'''
import sys
import os
import math
from FileProcessing import FileProcessing,Collection,Document
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
from nltk.tbl import feature
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
import re

class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []
        self.frequency = 0

    def append(self, pos):
        self.positions.append(pos)

    def term_freq(self):
        ''' return the term frequency in the document'''
        #ToDo
        self.frequency += 1
        return self.frequency

class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        self.sorted_postings= [] # may sort them by docID for easier query processing

    def add(self, docid, pos):
        ''' add a posting'''
        
        self.posting[docid].positions.append(pos)
        

class InvertedIndex:
    '''Creating Inverted index file with feature id for given news group data'''
  
    def __init__(self):
        self.items = {} # list of IndexItems
        self.nDocs = 0  # the number of indexed documents
        self.termfreq = {} #term frequency of each word


    def indexDoc(self, doc): # indexing a Document object
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''

        # ToDo: indexing only title and body; use some functions defined in util.py
        # (1) convert to lower cases,
        # (2) remove stopwords,
        # (3) stemming
        '''
        if os.path.exists("index_file1.txt"):
            os.remove("index_file1.txt")
        '''

        writeFile = open("index_file1.txt","a")
        
        termDocFreq = open("termDocFreq.txt","a")

        #convert to lower case
        tokenizer = RegexpTokenizer(r'\w+') 
        #tokens = tokenizer.tokenize(Pdoc.body)
            
        word_tokens = tokenizer.tokenize(doc.subject + " " + doc.body)
        word_tokens = [token.lower() for token in word_tokens]
        docid = doc.docID
        'writing total word count to termDocfreq file'
        #Tokens = word_tokenize(doc.subject + " " + doc.body)
        termDocFreq.write(docid + " " + str(len(word_tokens)) + "\n")

        #removing stopwords

        word_list = []
        stopWords = set(stopwords.words('english'))

        for w in word_tokens:
            if w not in stopWords:
                word_list.append(w)


        #stemming
        stem_list = []
        pos = 1
        ps = PorterStemmer()
        
        for w in word_list:
            w1 = ps.stem(w)
            stem_list.append(w1)

        #getting non redundant set of words
        nonRlist = set(stem_list)

        for w in nonRlist:
            #print(docid)
            self.items[w] = IndexItem(w)
            self.items[w].posting[docid] = Posting(docid)
            

        for w in stem_list:
            self.items[w].term = w                
            self.items[w].add(docid, pos)
            self.termfreq[w] = self.items[w].posting[docid].term_freq()
            #self.items
            pos += 1

        for key, value in self.items.items():
            writeFile.write(key + " " + docid + " " + str(value.posting[docid].positions)) #+ " " + str(self.termfreq[key]))
            writeFile.write("\n")

        self.items = {}


            
    def sort(self):
        ''' sort all posting lists by docID'''
        #ToDo
        
        if os.path.exists("index_file3.txt"):
            os.remove("index_file3.txt")
        file = open("index_file1.txt", "r")
        file1 = open("index_file3.txt", "a")
        lines = file.readlines()
        self.indexterms = {}
        self.indexterms1 = {}
        self.id_terms={}
        words = []
        

        for line in lines:
            line = line.replace('\n','')
            words = line.split(" ",2)
            self.indexterms1[words[0]] = []
                

        nonR = set(self.indexterms1.keys())

        for x in nonR:
            self.indexterms[x] = {}
            

        for line in lines:
            line = line.replace('\n','')
            words = line.split(" ",2)
            self.indexterms[words[0]].update( {words[1] : words[2]} )

        feature_id = 1

        for i in range(1,len(self.indexterms.keys())):
            self.id_terms[i] = {}

        for word in self.indexterms.keys():
            file1.write(str(feature_id) + " " + word + " " + str(self.indexterms[word])) #+ " " + self.indexdocid[self.indexterms[word]])
            self.id_terms[feature_id] = {word:self.indexterms[word]}
            feature_id += 1
            

    def find(self, term):
        return self.items[term]

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        global json1
        if os.path.exists(filename):
            os.remove(filename)
        json1 = json.dumps(self.id_terms,indent = 4)
        f = open(filename,"a")
        f.write(json1)

    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        #ToDo: return the IDF of the term

    # more methods if needed


def test():
    ''' test your code thoroughly. put the testing cases here'''
    print('Pass')

def indexingNewsGroupFiles():
    #ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python feature-extract.py directory_of_newsgroups_data feature_definition_file class_definition_file training_data_file"
    # the features are saved to feature_definition_file

    #doc = sys.argv[1]
    obj = InvertedIndex()

    'Removing file if already exist'
    
    if os.path.exists("index_file1.txt"):
            os.remove("index_file1.txt")
            
    if os.path.exists("termDocFreq.txt"):
        os.remove("termDocFreq.txt")
        
    filenames=os.listdir(rootdir)
    document=[]
    'iterating through the newsgroup directory and processing each documents in subdir'
    for subdir,dirs,files in os.walk(rootdir):
        for i,file in enumerate(files):
            if file !='.DS_Store':
                #print(subdir+"/"+file)
                ' FileProcessing class object where document title and body are processed'
                FP = FileProcessing(subdir+"/"+file)
                for doc in FP.docs:
                    obj.indexDoc(doc)
                    #print("Creating feature Index for document",doc.docID)
                #document.append(FP.docs)

    #for doc in FP.docs:
#        obj.indexDoc(doc)

    #sorting
    obj.sort()
    
    #writeFile = open(sys.argv[2],"a")
    obj.save(feature_def_file)
            
    print('Done')
def classLabels():
    'creating class labels by traversing through directory'
    classLabel ={}
    if os.path.exists(class_def_file):
            os.remove(class_def_file)
    classDefFile = open(class_def_file,"a")   
    for f in filenames:
        if os.path.isdir(os.path.join(os.path.abspath(rootdir), f)):
            if 'comp.' in f:
                if int(0) not in classLabel:
                    classLabel[int(0)]=[f]
                else:
                    classLabel[int(0)].append(f)
            elif 'rec.' in f:
                if int(1) not in classLabel:
                    classLabel[int(1)]=[f]
                else:
                    classLabel[int(1)].append(f)
            elif 'sci.' in f:
                if int(2) not in classLabel:
                    classLabel[int(2)]=[f]
                else:
                    classLabel[int(2)].append(f)
            elif 'misc.' in f:
                if int(3) not in classLabel:
                    classLabel[int(3)]=[f]
                else:
                    classLabel[int(3)].append(f)
            elif 'talk.politics.' in f:
                if int(4) not in classLabel:
                    classLabel[int(4)]=[f]
                else:
                    classLabel[int(4)].append(f)
            else:
                if int(5) not in classLabel:
                    classLabel[int(5)]=[f]
                else:
                    classLabel[int(5)].append(f)

    for k,v in classLabel.items():
        for i in v:
            classDefFile.write("%d %s\n" %(k ,i))
'returns the class label when given the document id'
def findFile(documentid):
        folderName =''
        for subdir,dirs,files in os.walk(rootdir):
            for file in files:
                if (file==documentid):
                    #print(file)
                    path = os.path.normpath(subdir)
                    #print(path.split(os.sep))
                    folderName=path.split(os.sep)[-1]
                    #print(subdir)
                    #print(folderName)
        return folderName
'''Creates training definition file for each of the feature categeory like
    term frequency,Inverse Document frequency and TD-IDF'''
def trainingDataset():
    feature_Dict ={}
    #featureid_term ={}
    document_Label={}
    class_Label={}
    termfiletext="term_featureidfile.txt"
    document_Labelfile="document_Label.txt"
    classDefinition=open(class_def_file,"r")
    for x in classDefinition:
        folder=x.split()[1]
        label=x.split()[0]
        class_Label[folder]=label
    #print(class_Label)
        
    featureDefinition=load(feature_def_file)
    if os.path.exists(termfiletext):
            os.remove(termfiletext)
    termfile=open(termfiletext,"a+")
    if os.path.exists(document_Labelfile):
            os.remove(document_Labelfile)
    DL_file=open(document_Labelfile,"a+")
    print("Creating training data file for terms is in progress")
    for featureid in featureDefinition:
        #print(featureid,featureDefinition[featureid])
        featureid_Dic=featureDefinition[featureid]
        #print(featureid_Dic)
        for terms,docids in featureid_Dic.items():
                documentid ={}
                #print(featureid,terms,docids)
                for docid,pos in docids.items():
                    #print(docid)
                    #print(featureid,docid,pos,len(json.loads(pos)))
                    f1 = open("termDocFreq.txt","r")
                    N = 0
                    lines = f1.readlines()
                    for line in lines:
                        docID = line.split()[0]
                        #print(type(docid))
                        if docid == docID:
                            N = int(line.split()[1])
                            #print(N)
                            break
                    TF=len(json.loads(pos))
                    featureid=int(featureid)
                    #id=("%d:%d" %(featureid ,TF))
                    if type_feature=='0':
                        documentid[int(docid)]=TF
                    elif type_feature == '1': #
                        IDF = 1+math.log10(1954/len(docids.items()))
                        documentid[int(docid)]=IDF
                    elif type_feature == '2':
                        #TF=len(json.loads(pos))/N
                        IDF = 1+math.log10(1954/len(docids.items()))
                        documentid[int(docid)]=(TF/N)*IDF
                    #featureid=int(featureid)
                    DL_file.write("%d %s\n" %(int(docid),findFile(docid)))
                #print("dic",documentid)
                if int(featureid) not in feature_Dict:
                    feature_Dict[int(featureid)]=documentid
                else:
                    feature_Dict[int(featureid)].append(documentid)
                #featureid_term[featureid]=terms
                #f_t=featureid+" "+terms
                #print(featureid,terms,"\n")
                #print(document_Label)
                termfile.write("%d %s\n" %(int(featureid),terms))
                #print(feature_Dict)
                print("Feature id",featureid)
    termfile.close()
    DL_file.close()
    
    #print(feature_Dict)
    #print(featureid_term)
    doc_featureMatrix=pd.DataFrame(feature_Dict)
    doc_featureMatrix=doc_featureMatrix.fillna(0)
    feature_documentMatrix=doc_featureMatrix
    #feature_documentMatrix=feature_documentMatrix.rename_axis("Documentid↓:index->", axis="columns")
    new_index=[]
    if os.path.exists("pdindex"):
            os.remove("pdindex")
    t=open("pdindex","a+")
    for index1 in feature_documentMatrix.index:
        file=findFile(str(index1))
        label=class_Label[file]
        new_index.append(label)
        writeto=str(index1)+" "+label
        t.write(writeto+"\n")
    #print(new_index)
    #print(df.index)
    #print(df)
    #feature_documentMatrix=feature_documentMatrix.assign(D_LABEL=new_index)
    feature_documentMatrix.insert(0, "ClassLabel", new_index) 
    #feature_documentMatrix=feature_documentMatrix.reindex(new_index)
    #feature_documentMatrix=feature_documentMatrix.set_index('D_LABEL')
    #print(feature_documentMatrix)
    
    X=feature_documentMatrix.loc[:, feature_documentMatrix.columns != 'ClassLabel']
    y=feature_documentMatrix["ClassLabel"]
    'converting dataframes to libsvm format'
    if type_feature == '0':
        if os.path.exists(training_data_file_TF):
            os.remove(training_data_file_TF)
        dump_svmlight_file(X,y,training_data_file_TF)
        print(training_data_file_TF,"is created")
    elif type_feature == '1':
        if os.path.exists(training_data_file_IDF):
            os.remove(training_data_file_IDF)
        dump_svmlight_file(X,y,training_data_file_IDF)
        print(training_data_file_IDF,"is created")
    elif type_feature == '2':
        if os.path.exists(training_data_file_TFIDF):
            os.remove(training_data_file_TFIDF)
        dump_svmlight_file(X,y,training_data_file_TFIDF)
        print(training_data_file_TFIDF,"is created")

        
    '''        
    print("finaltds",tds.keys(),tds.items())             
    if os.path.exists(training_data_file):
            os.remove(training_data_file)
    with open(training_data_file,'w') as output_file:
            json.dump(tds,output_file,indent=3)
            output_file.close()
'''
    
def save(obj,filename):
        ''' save to disk :reusable function to save file to disk using JSON'''
        # ToDo: using your preferred method to serialize/deserialize the index
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename,'w') as output_file:
            json.dump(obj,output_file,indent=3)
            output_file.close()

def load(filename):
        ''' load from disk'''
        'reusable function to Load saved JSON file from disk'
        o_filename = open(filename)
        input = json.load(o_filename)
        return input
def dfread():
    #dic={"1":{"11":12},{"12":22}],"2":[{"21":2},{"22":2}]}
    class_Label={}
    new_index=[]
    classDefinition=open(class_def_file,"r")
    for x in classDefinition:
        folder=x.split()[1]
        label=x.split()[0]
        class_Label[folder]=label
    dic={11: {61022: 1}, 22: {76410: 1}, 33: {103758: 2, 59261: 3}, 44: {67346: 1}}
    doc_featureMatrix=pd.DataFrame(dic)
    doc_featureMatrix=doc_featureMatrix.fillna(0)
    feature_documentMatrix=doc_featureMatrix
    #feature_documentMatrix=feature_documentMatrix.rename_axis("Documentid↓:index->", axis="columns")
    t=open("pdindex.text","a+")
    for index1 in feature_documentMatrix.index:
        file=findFile(str(index1))
        label=class_Label[file]
        print(index1,file,int(label))
        new_index.append(int(label))
    print(new_index)
    print(feature_documentMatrix.index)
    print(feature_documentMatrix)
    feature_documentMatrix.insert(0, "ClassLabel", new_index) 
    #feature_documentMatrix=feature_documentMatrix.assign(Doclabel=new_index)
    #feature_documentMatrix=feature_documentMatrix.reindex(new_index)
    #feature_documentMatrix=feature_documentMatrix.set_index('Doclabel')
    #feature_documentMatrix=feature_documentMatrix.rename_axis("Documentid↓:index->", axis="columns")
    print(feature_documentMatrix)
    #print(feature_documentMatrix.head())
    if os.path.exists("rows.txt"):
            os.remove("rows.txt")
    rows=open("rows.txt","a+")
    for keys in feature_documentMatrix.index:
        print(keys,np.array(feature_documentMatrix.loc[keys]))
        #rows.to_csv(feature_documentMatrix.loc[keys])
        #feature_documentMatrix.loc[keys].to_csv(rows, index=True)
        for i in np.array(feature_documentMatrix.loc[keys]):
            rows.write(str(i)+" ")

        rows.write("\n")
    #file=pd.read_csv(training_data_file,index='name',delimiter="\t")
    #print(file)
    X=feature_documentMatrix.loc[:, feature_documentMatrix.columns != 'ClassLabel']
    y=feature_documentMatrix["ClassLabel"]
    dump_svmlight_file(X,y,"raw.txt")
    training_data_file = "raw.txt"
    feature_vectors, targets = load_svmlight_file(training_data_file)
    print(feature_vectors,"targtes",targets)
'list of system arguments defintions /assignemnts' 
rootdir = sys.argv[1]#'mini_newsgroups'
feature_def_file = sys.argv[2]#"feature_definition_file.json"
class_def_file = sys.argv[3]#"class_definition_file"
if sys.argv[5]=='0':
    training_data_file_TF = sys.argv[4]+".TF"#"training_data_file.TF"
if sys.argv[5]=='1':
    training_data_file_IDF=sys.argv[4]+".IDF" #"training_data_file.IDF"
if sys.argv[5]=='2':
    training_data_file_TFIDF=sys.argv[4]+".TFIDF" #"training_data_file.TFIDF"
type_feature = sys.argv[5] #0 for tf,1 for idf,2 for tfidf
if __name__ == '__main__':
    #test()
    filenames=os.listdir(rootdir)
    if not os.path.exists(feature_def_file):
            indexingNewsGroupFiles()
            print("Feature definition file creation in progress")
    classLabels()
    trainingDataset()
