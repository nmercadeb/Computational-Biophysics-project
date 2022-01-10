# 
#         COMBIO PROJECT: DATA ANALISIS
#         Marta Alcalde & Núria Mercadé     
# 

# ===========================================================================
#    PREDICCTION MODELS - SUPORT VECTOR MACHINE
# ===========================================================================
from sklearn.model_selection       import train_test_split
from sklearn.model_selection       import RepeatedStratifiedKFold
from sklearn.model_selection       import cross_val_score
from sklearn.metrics               import confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing         import StandardScaler
from sklearn.svm                   import SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.linear_model          import LogisticRegression
#from sklearn.tree                  import DecisionTreeClassifier
#from sklearn.neighbors             import KNeighborsClassifier
import xlsxwriter
import seaborn           as sns
import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np


# Set seed
seed = 90000
# Read Excel
file = "C:/Users/Marta/Dropbox/COMBIO/dades.xlsx"
# file = "/Users/nmercade/Desktop/dades.xlsx"

data_dirty   = pd.read_excel(file,header = 0, usecols = "A,B,H:AD")
# Delete one video # 120, 221, 447
data = data_dirty.drop([0,1,118,219,342,445])
data = data.loc[:,("Test","% Infarct","Moving_time(%)","vm_nose(cm/s)","vm_butt",\
                "vm_tail_end","vm_pawE","vm_pawD","vm_pawe","vm_pawd",\
                "d_nose-but")];

mydataTR = {0: data.loc[(data["Test"] == "TR - POST - 24H") | (data["Test"] == "TR - POST - 48H") |\
                  (data["Test"] == "TR - POST - 72H") | (data["Test"] == "TR - PRE")],\
          1: data.loc[(data["Test"] == "TR - POST - 24H") | (data["Test"] == "TR - POST - 48H") |\
                  (data["Test"] == "TR - POST - 72H")],\
          2: data.loc[(data["Test"] == "TR - PRE")], 3: data.loc[(data["Test"] == "TR - POST - 24H")],\
          4: data.loc[(data["Test"] == "TR - POST - 48H")], 5: data.loc[(data["Test"] == "TR - POST - 72H")], \
          6: data.loc[(data["Test"] == "TR - POST - 48H") | (data["Test"] == "TR - POST - 72H")]};


datTR = {0: 'ALL', 1: 'ALL POST', 2: 'PRE', 3: 'POST 24H', 4: 'POST 48H', 5: 'POST 72H', 6: 'POST 48H & 72H'}


for jj in range(7):
    data = mydataTR[jj];
    print('The dataset is {}'.format(datTR[jj]))
    print(jj)
    
    # Define predictor and response variables
    # 1. Response variable:
        # 0: Not having an ictus
        # 1: Having an ictus
    inf = np.array(data["% Infarct"].dropna()); y = [];
    for i in range(len(inf)):
        if inf[i] > 0:
            y.append(1)
        else:
            y.append(0)
    # 2. Independent variables
    x = data.loc[:,"Moving_time(%)":"d_nose-but"];

    # Creation of two datasets: test and train
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size = 0.25, random_state = seed)

    # Visualization of the response variable to prove if it is balanced.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.hist(Ytrain)

    # Scale the feature to expect normal distribution of the data.
    sc = StandardScaler()
    XtrainN = sc.fit_transform(Xtrain)
    XtestN  = sc.transform(Xtest)

    # =========================================================================
    # SVM - rbf KERNEL
    # =========================================================================
    model = SVC(kernel = 'rbf', random_state = seed)
    model.fit(XtrainN,Ytrain)

    #Prediction of our test data
    Ypred = model.predict(XtestN)

    # KFold method
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10, random_state = seed) #cv = StratifiedKFold(n_splits = 10)
    # Evaluate model
    scores = cross_val_score(model, Xtrain, Ytrain, scoring = 'accuracy', cv = cv)
    print('Train accuracy: {}'.format(np.mean(scores)*100))

    # Confusion matrix for the rbf kernel
    cm = confusion_matrix(Ytest,Ypred)
    # Test accuracy 
    acc = (cm[0,0]+cm[1,1])/sum(sum(cm))*100
    print('Test accuracy(linear kernel): {}'.format(acc))
    # Sensitivity
    sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])*100
    print('Sensitivity(linear kernel): The {} were correctly identified as not having an ictus'.format(sensitivity))
    # Specificity
    specificity = cm[1,1]/(cm[1,1]+cm[0,1])*100
    print('Specificity(linear kernel): The {} were correctly identified as having an ictus'.format(specificity))
    

    sns.set(style = 'white')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(np.eye(2), annot = cm, fmt = 'g', annot_kws = {'size': 30},
                cmap = sns.color_palette(['lightcoral', 'darkseagreen'], as_cmap=True), cbar=False,
                yticklabels=['Negative', 'Positive'], xticklabels=['Negative', 'Positive'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelsize=19, length = 0)
    ax.set_xlabel('Predicted Values', size = 20)
    ax.set_ylabel('Actual Values', size = 20)
    additional_texts = ['(True negative)', '(False positive)', '(False negative)', '(True positive)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
                ha='center', va='top', size=24)
    plt.tight_layout()
    path = '.\\SVM\\CM_{}'.format(jj)
    fig.savefig(path)
    
    # =========================================================================
    # ACCURACY RATA
    # =========================================================================
    # Read the identification of the rat
    rat = pd.read_excel(file,header = 0, usecols = "A,B,H:AD");
    rat = rat.Mouse;
    # Select those identifications that are from the test part
    ratTest  = rat.iloc[Xtest.index];
    # Vector that contains each identification of the rat ones.
    lrt = list(set(ratTest));

    #convert each Series to a DataFrame (it is easier to operate with them)
    rt_df = ratTest.to_frame(name = 'Id');
    yp_df = pd.DataFrame(Ypred, columns = ['Pred']); yp_df.index = Xtest.index;
    yt_df = pd.DataFrame(Ytest, columns = ['Test']); yt_df.index = Xtest.index;

    Rpred = []; Rtest = [];
    ix1 = 0;
    
    # Loop to gather predictions and identification of the rat
    for ix in lrt:
        # Prediction
        pos = yp_df[ratTest == lrt[ix1]]
        ones = (pos.values == 1).sum()
        zero = (pos.values == 0).sum()
        
        if ones > zero:
            Rpred.append(1)
        if ones < zero:
            Rpred.append(0)
        if ones == zero:
            Rpred.append(2)
        
        # Test data
        pos = yt_df[ratTest == lrt[ix1]]
        ones = (pos.values == 1).sum()
        if ones >= 1:
            Rtest.append(1)
        else:
            Rtest.append(0)

        ix1 = ix1 + 1
    
    # Accuracy of rat
    cm = confusion_matrix(Rtest,Rpred)
    # Test accuracy 
    acc = (cm[0,0]+cm[1,1])/sum(sum(cm))*100       
    
    print('Accuracy of rat', acc)
    print(' ')
    

    
# ===========================================================================
# SVM - POLYNOMIAL KERNEL
# ===========================================================================
# model = SVC(kernel = 'poly', random_state = seed)
# model.fit(XtrainN, Ytrain)
# Ypred1 = model.predict(XtestN)

# # Confusion matrix for the polinomial kernel
# cm = confusion_matrix(Ytest,Ypred1)
# # Test accuracy 
# acc = (cm[0,0]+cm[1,1])/sum(sum(cm))*100
# print('Test accuracy(poly kernel): {}'.format(acc))
# # Sensitivity
# sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])*100
# print('Sensitivity(poly kernel): The {} were correctly identified as not having an ictus'.format(sensitivity))
# # Specificity
# specificity = cm[1,1]/(cm[1,1]+cm[0,1])*100
# print('Specificity(poly kernel): The {} were correctly identified as having an ictus'.format(specificity))
# print(' ')

# ===========================================================================
# SVM - RBF KERNEL
# ===========================================================================
# model = SVC(kernel = 'rbf', random_state = seed)
# model.fit(XtrainN, Ytrain)
# Ypred1 = model.predict(XtestN)

# # Confusion matrix for the polinomial kernel
# cm = confusion_matrix(Ytest,Ypred1)
# # Test accuracy 
# acc = (cm[0,0]+cm[1,1])/sum(sum(cm))*100
# print('Test accuracy(RBF kernel): {}'.format(acc))
# # Sensitivity
# sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])*100
# print('Sensitivity(RBF kernel): The {} were correctly identified as not having an ictus'.format(sensitivity))
# # Specificity
# specificity = cm[1,1]/(cm[1,1]+cm[0,1])*100
# print('Specificity(RBF kernel): The {} were correctly identified as having an ictus'.format(specificity))
# print(' ')

# ===========================================================================
# SVM - SIGMOID KERNEL
# ===========================================================================
# model = SVC(kernel = 'sigmoid', random_state = seed)
# model.fit(XtrainN, Ytrain)
# Ypred1 = model.predict(XtestN)

# # Confusion matrix for the polinomial kernel
# cm = confusion_matrix(Ytest,Ypred1)
# # Test accuracy 
# acc = (cm[0,0]+cm[1,1])/sum(sum(cm))*100
# print('Test accuracy(sigmoid kernel): {}'.format(acc))
# # Sensitivity
# sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])*100
# print('Sensitivity(sigmoid kernel): The {} were correctly identified as not having an ictus'.format(sensitivity))
# # Specificity
# specificity = cm[1,1]/(cm[1,1]+cm[0,1])*100
# print('Specificity(sigmoid kernel): The {} were correctly identified as having an ictus'.format(specificity))
