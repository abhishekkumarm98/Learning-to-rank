from sklearn.cluster import KMeans
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
%matplotlib inline

maxAcc = 0.0
maxIter = 0
C_Lambda = 8
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 10
PHI = []
IsSynthetic = False

def GetTargetVector(filePath): # To get output in vector form
    t = []
    with open(filePath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t

def GenerateRawData(filePath, IsSynthetic):   # To get input features in refined and matrix form 
    dataMatrix = [] 
    with open(filePath, newline='') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    if IsSynthetic == False :# As we know that boolean values have value in python, I mean for False, it gives 0 and for True,1
        dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1) # That is the reason IsSynthetic Variable is assigned False
    dataMatrix = np.transpose(dataMatrix)  # By checking the condition, we will remove those columns which contain zeros only     
    #print ("Data Matrix Generated..")
    return dataMatrix  # It will give transpose of dataMatrix of shape (41,69623)
    
    
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): # Creating the training target variable(80% of target variable)
    TrainingLen = math.ceil(len(rawTraining)*(TrainingPercent*0.01))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t # (55699,)

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80): # Making the Training dataMatrix which is 80% of input dataset
    T_len = math.ceil(len(rawData[0])*0.01*TrainingPercent)
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2 # (41,55699)

def GenerateValData(rawData, ValPercent, TrainingCount): # Creating Validation dataMatrix(10% of input dataset)
    valSize = math.ceil(len(rawData[0])*ValPercent*0.01)
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix # (41,6962)

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): # Making the Validation target(10% of input dataset)
    valSize = math.ceil(len(rawData)*ValPercent*0.01)
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t  # (6962,)

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,TrainingLen):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]  # BigSigma is getting assigned diagonally by all values of varVect
    if IsSynthetic == True: 
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    
    return BigSigma # It will return the size (41,41) of BigSigma 
    
    
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data) 
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    PHI = np.zeros((TrainingLen,len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma) 
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI # Its shape is (55699,10)
    

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0])) # (10,10)
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W # (10,)
    
def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct): # rms = root mean square
    sum = 0.0
    accuracy = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))
    
# Fetch and Prepare Dataset
RawTarget = GetTargetVector('Querylevelnorm_t.csv')
RawData   = GenerateRawData('Querylevelnorm_X.csv',IsSynthetic)

# Prepare Training Data
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)

# Prepare Validation Data
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)

# Prepare Test Data
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(TestDataAct.shape)
print(TestData.shape)

k_list = [2*M for M in range(1,21)]
WCSS = [] # Within cluster sum of square
for M in k_list:
    kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
    WCSS.append(kmeans.inertia_)

# Elbow Method
plt.plot(k_list,WCSS,'o-')
plt.xlabel("Number of clusters M----->")
plt.ylabel("Within cluster sum of square----->")
plt.title("WCSS Vs. Number of clusters M")

# Closed Form Solution [Finding Weights using Moore- Penrose pseudo- Inverse Matrix]
kmeans = KMeans(n_clusters=9, random_state=0).fit(np.transpose(TrainingData))

Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)

print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)

# To choose a value of regularization parameter(Lamda) with training set

Lamda=[lamda*0.42 for lamda in range(1,21)]
Training_Accuracy = []
for lamda in Lamda:
  Weight = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,lamda) 
  TR_TEST_OUT  = GetValTest(TRAINING_PHI,Weight)
  TrainingAcc= str(GetErms(TR_TEST_OUT,TrainingTarget))
  Training_Accuracy.append(float(TrainingAcc.split(',')[0]))
plt.plot(Lamda,Training_Accuracy,'ro-')
plt.ylabel("Training_Accuracy")
plt.xlabel("Lamda")
plt.title("Training Accuracy Vs. Lamda")

# To select a value of lamda with validation data

Lamda=[l*0.03 for l in range(1,31)]
ValidationAcc = []
for lamda in Lamda:
  w_= GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,lamda) 
  VAL_TEST_OUT = GetValTest(VAL_PHI,w_)
  Validation_Accuracy= str(GetErms(VAL_TEST_OUT,TrainingTarget))
  ValidationAcc.append(float(Validation_Accuracy.split(',')[0]))
plt.plot(Lamda,ValidationAcc,'r*-')
plt.ylabel("Validation Accuracy----->")
plt.xlabel("Lamda----->")
plt.title("Validation Accuracy Vs. Lamda")


# Finding Erms on training, validation and test set
TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))

print ("------------------LeToR Data------------------------")
print ('----------------------------------------------------')
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------------------------------------------')
print ("M = 10 \nLambda = 0.03\n")

print("Accuracy")
print("Training accuracy   = " + TrainingAccuracy.split(',')[0])
print("Validation accuracy = " +  ValidationAccuracy.split(',')[0])
print("Test accuracy       = " +  TestAccuracy.split(',')[0]+"\n")

print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))

# Gradient Descent solution
W_Now        = np.dot(220, W) # It intializes with random value
La           = 2 # Lamda
learningRate = 0.01
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,400):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))
    
    
# To find the learning rate 
Learningrate = [0.009*i for i in range(1,19)]
Training_Erms=[]
for l in Learningrate:
    
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(l,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    Training_Erms.append(float(Erms_TR.split(',')[1]))

    
plt.plot(Learningrate,Training_Erms,'ro-')
plt.ylabel("Training_Erms")
plt.xlabel("Learningrate")
plt.title("Training_Erms Vs. Learningrate")

print ('----------Gradient Descent Solution--------------------')
print("Accuracy")
print("Training accuracy   = " +TrainingAccuracy.split(',')[0])
print("Validation accuracy = " + ValidationAccuracy.split(',')[0])
print("Test accuracy       = " + TestAccuracy.split(',')[0]+"\n")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
