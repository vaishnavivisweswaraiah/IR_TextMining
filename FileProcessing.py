'''

processing the special format used by the Cranfield Dataset



'''
import os 
import json
from builtins import input

class Document:
    def __init__(self, docid, subject,body):
        self.docID = docid
        self.subject = subject
        self.body = body


    # add more methods if needed


class Collection:
    ''' a collection of documents'''

    def __init__(self):
        self.docs = {} # documents are indexed by docID
        #print("Insidecollectionclass from  doc",self.docs)

    def find(self, docID):
        ''' return a document object'''
        #if self.docs.has_key(docID):
        #print(self.docs,docID)
        if docID in self.docs:
            return self.docs[docID]
        else:
            return None

    # more methods if needed
def test():
    folderinfo ={}
    classinfo={}
    class_def=classlabel()
    Invertedclass_def={value:key for key,values in class_def.items() for value in values}
    classlabelfiles={}
    #print(Invertedclass_def)
    docid=[]
    for subdir,dirs,files in os.walk(rootdir):
        for file in files:
            if file !='.DS_Store':
                fname=subdir.split('/')[-1]
                folderinfo[fname]=len(files)
                #print(file)
                docid.append(int(file))
    
    for filekeys,labels in Invertedclass_def.items():
      if labels not in classlabelfiles:
          classlabelfiles[labels]=[folderinfo[filekeys]] 
      else:
          classlabelfiles[labels].append(folderinfo[filekeys])
    #print(classlabelfiles)
    for labels,filecount in classlabelfiles.items():
        classlabelfiles[labels]=sum(filecount)
    
    print(folderinfo)
    print("number of folders",len(folderinfo.values()))
    print("Total number of files",sum(folderinfo.values()))
    print(classlabelfiles)
    print("total number of files",sum(classlabelfiles.values()))
    o_filename = open('pdindex')
    r=o_filename.readlines()
    #Politics,Rec,Computer,Religion,Science,Miscellaneous
    C=0
    P=0
    R=0
    Rl=0
    S=0
    M=0
    d=[]
    for x in r :
        if 'Computer' in x:
            C=C+1
            #print(x)
            #print(C)
        elif 'Politics'in x:
            P+=1
        elif 'Rec' in x:
            R+=1
        elif 'Religion' in x:
            Rl+=1
        elif 'Science' in x:
            S+=1
        elif 'Miscellaneous' in x:
            M+=1
        d.append(int(x.split()[0]))
        #print(x)
    print(P,R,C,Rl,S,M)
    print(len(d),d)
    print(len(docid),sorted(docid))
    diff=set(d).intersection((set(docid)))
    print(set(docid).difference(diff))
    
     
    

class FileProcessing:
    def __init__(self, filename):
        self.docs = []

        cf = open(filename,encoding="ISO-8859-1")
        #print(filename.split('/')[-1])
        docid = filename.split('/')[-1]
        #print(type(docid))
        subject = ''
        body = ''
        
        for line in cf:
            if 'Subject:' in line:
                if 'Re:' in line:
                    subject=line.split('Re:')[1]
                else:
                    subject=line.split('Subject:')[1]
                # start a new document
            elif 'Lines:' in line:
                #print(line)
                try:
                    #print(type(line.split('Lines:')[1]))
                    lines=int(line.split('Lines:')[1])
                    #print(lines)
                    for eachline in cf.readlines()[-lines:]:
                        body=body+eachline
                    
                except Exception as e:
                    #print(e)
                    #print(cf.readlines())
                    for eachline in cf.readlines():
                        body=body+eachline
                    #print(body)
                
        #if docid=='51126':        
        #    print(body)
        self.docs.append(Document(docid, subject,body)) # the last one
        #print(self.docs)
if __name__ == '__main__':
    ''' testing '''
    rootdir='mini_newsgroups'
    filenames=os.listdir(rootdir)
    document=[]
    for subdir,dirs,files in os.walk(rootdir):
        for i,file in enumerate(files):
            if file !='.DS_Store':
                #print(subdir+"/"+file)
                FP = FileProcessing(subdir+"/"+file)
   
    
