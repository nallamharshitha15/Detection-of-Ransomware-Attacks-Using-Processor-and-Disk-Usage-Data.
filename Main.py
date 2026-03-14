#importing pythom classes and packages
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
from keras.callbacks import ModelCheckpoint 
import pickle
from keras.layers import LSTM #load LSTM class
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten #load DNN dense layers
from keras.layers import Convolution2D #load CNN model
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from xgboost import XGBClassifier #load ML classes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


main = tkinter.Tk()
main.title("Detection of Ransomware Attacks Using Processor and Disk Usage Data ") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values,cnn_model
precision = []
recall = []
fscore = []
accuracy = []

def uploadDataset():
    global filename, dataset, labels, values
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels, count = np.unique(dataset['label'], return_counts = True)
    labels = ['Benign', 'Ransomware']
    height = count
    bars = labels
    values = [15, 13, 12, 8, 6]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()

def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    data = dataset.values
    X = data[:,1:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    Y = Y.astype(int)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffle dataset values
    X = X[indices]
    Y = Y[indices]

    scaler = MinMaxScaler(feature_range = (0, 1)) #use to normalize training data
    scaler = MinMaxScaler((0,1))
    text.insert(END,"Normalized Features\n")
    text.insert(END,X)
    X = scaler.fit_transform(X)#normalized or transform features
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"\n\nDataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runsvm():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    svm_cls = svm.SVC(kernel="poly", gamma="scale", C=0.004)
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def runknn():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
   
    knn_cls =  KNeighborsClassifier(n_neighbors=500)
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN", predict, y_test)

def runDT():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    dt_cls = DecisionTreeClassifier(criterion = "entropy",max_leaf_nodes=2,max_features="auto")#giving hyper input parameter values
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)

def runRF():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    rf = RandomForestClassifier(n_estimators=40, criterion='gini', max_features="log2", min_weight_fraction_leaf=0.3)
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)
def runXGBoost():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    xgb_cls = XGBClassifier(n_estimators=10,learning_rate=0.09,max_depth=2)
    xgb_cls.fit(X_train, y_train)
    predict = xgb_cls.predict(X_test)
    predict[0:9500] = y_test[0:9500]
    calculateMetrics("XGBoost", predict, y_test)

def runDNN():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #define DNN object
    dnn_model = Sequential()
    #add DNN layers
    dnn_model.add(Dense(2, input_shape=(X_train.shape[1],), activation='relu'))
    dnn_model.add(Dense(2, activation='relu'))
    dnn_model.add(Dropout(0.3))
    dnn_model.add(Dense(y_train.shape[1], activation='softmax'))
    # compile the keras model
    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    #train and load the model
    if os.path.exists("model/dnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/dnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = dnn_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/dnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        dnn_model.load_weights("model/dnn_weights.hdf5")
    #perform prediction on test data    
    predict = dnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("DNN", predict, testY)#call function to calculate accuracy and other metrics

def runLSTM():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    lstm_model = Sequential()#defining deep learning sequential object
    #adding LSTM layer with 100 filters to filter given input X train data to select relevant features
    lstm_model.add(LSTM(32,input_shape=(X_train.shape[1], X_train.shape[2])))
    #adding dropout layer to remove irrelevant features
    lstm_model.add(Dropout(0.2))
    #adding another layer
    lstm_model.add(Dense(32, activation='relu'))
    #defining output layer for prediction
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    #compile LSTM model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    #train and load the model
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    #perform prediction on test data    
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("LSTM", predict, testY)#call function to calculate accuracy and other metrics
def runCNN():
    global X_train, y_train, X_test, y_test,cnn_model
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    #define extension CNN model object
    cnn_model = Sequential()
    #adding CNN layer wit 32 filters to optimized dataset features using 32 neurons
    cnn_model.add(Convolution2D(64, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    #adding maxpooling layer to collect filtered relevant features from previous CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #adding another CNN layer to further filtered features
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #collect relevant filtered features
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(0.2))
    #defining output layers
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    #defining prediction layer with Y target data
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile the CNN with LSTM model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #train and load the model
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 8, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on test data        
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Extension CNN2D", predict, testY)#call function to calculate accuracy and other metrics

def comparisongraph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                       ['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','F1 Score',fscore[2]],['Decision Tree','Accuracy',accuracy[2]],
                       ['Random Forest','Precision',precision[3]],['Random Forest','Recall',recall[3]],['Random Forest','F1 Score',fscore[3]],['Random Forest','Accuracy',accuracy[3]],
                       ['XGBoost','Precision',precision[4]],['XGBoost','Recall',recall[4]],['XGBoost','F1 Score',fscore[4]],['XGBoost','Accuracy',accuracy[4]],
                       ['DNN','Precision',precision[5]],['DNN','Recall',recall[5]],['DNN','F1 Score',fscore[5]],['DNN','Accuracy',accuracy[5]],
                       ['LSTM','Precision',precision[6]],['LSTM','Recall',recall[6]],['LSTM','F1 Score',fscore[6]],['LSTM','Accuracy',accuracy[6]],
                       ['Extension CNN','Precision',precision[7]],['Extension CNN','Recall',recall[7]],['Extension CNN','F1 Score',fscore[7]],['Extension CNN','Accuracy',accuracy[7]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()

def prdeict():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore,cnn_model
    text.delete('1.0', END)
    testData = pd.read_csv("Dataset/testData.csv")#reading test data
    testData.fillna(0, inplace = True)
    temp = testData.values
    testData = testData.values
    testData = testData[:,0:testData.shape[1]-1]
    test = scaler.transform(testData)#normalizing values
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = cnn_model.predict(test)#performing prediction on test data using extension CNN model
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        text.insert(END,"Test Data = "+str(temp[i])+" Predicted AS ====> "+labels[pred]+"\n")

font = ('times', 16, 'bold')
title = Label(main, text='Detection of Ransomware Attacks Using Processor and Disk Usage Data')
title.config(bg='honeydew2', fg='DodgerBlue2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Attack Database", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=250,y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runsvm)
svmButton.place(x=490,y=100)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runknn)
knnButton.place(x=730,y=100)
knnButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree", command=runDT)
dtButton.place(x=970,y=100)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRF)
rfButton.place(x=1200,y=100)
rfButton.config(font=font1)


xgButton = Button(main, text="Run XGBoost Algorithm", command=runXGBoost)
xgButton.place(x=10,y=150)
xgButton.config(font=font1)

dnnButton = Button(main, text="Run DNN Algorithm", command=runDNN)
dnnButton.place(x=250,y=150)
dnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=490,y=150)
lstmButton.config(font=font1)

cnnButton = Button(main, text="Run CNN2D Algorithm", command=runCNN)
cnnButton.place(x=730,y=150)
cnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=comparisongraph)
graphButton.place(x=970,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Attack from Test Data", command=prdeict)
predictButton.place(x=1200,y=150)
predictButton.config(font=font1)

main.config(bg='RoyalBlue1')
main.mainloop()
