## COMP7404 - Computational Intelligence and Machine Learning; Instructor: 
## Will employee leave the company or not? – Comparison of Machine Learning Techniques

## Main.ipynb

Github link: https://github.com/cuteyanna/COMP7404-Will-employee-leave-the-company-or-not

The Demo video also provides a lot of instrucitons.

This is the main code generating and comparing all the models we built.

A brief instruction of HR_EDA is also included in this README file.

### 1. Installing Python packages

Please make sure you have installed the following packages successfully. The version of sklearn and numpy should also be updated.
Then run every block to import the packages.


```python
!pip install deslib
!pip install sklearn
!pip install keras
!pip install tensorflow
!pip install matplotlib
```

    Requirement already satisfied: deslib in c:\users\chant\anaconda3\lib\site-packages (0.3.5)
    Requirement already satisfied: scikit-learn>=0.21.0 in c:\users\chant\anaconda3\lib\site-packages (from deslib) (0.24.2)
    Requirement already satisfied: numpy>=1.17.0 in c:\users\chant\anaconda3\lib\site-packages (from deslib) (1.20.3)
    Requirement already satisfied: scipy>=1.4.0 in c:\users\chant\anaconda3\lib\site-packages (from deslib) (1.7.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\chant\anaconda3\lib\site-packages (from scikit-learn>=0.21.0->deslib) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\users\chant\anaconda3\lib\site-packages (from scikit-learn>=0.21.0->deslib) (1.1.0)
    


```python
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Importing DS techniques
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des import METADES
from deslib.static import StackedClassifier
from deslib.util.datasets import make_P2
```


```python
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
from keras import losses
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, recall_score, f1_score
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
```


```python
#visualization
from matplotlib.cm import get_cmap
from matplotlib.ticker import FuncFormatter
```

### 2. Data Loading and Cleaning ( One hot encoding )

Please make sure that all of these 3 csv files are PLACED IN THE SAME DIRECTORY of the main.ipynb's. Nothing special, you only need to run every block subsequently.


```python
general_data = pd.read_csv('general_data.csv')
employee_survey_data = pd.read_csv('employee_survey_data.csv')
manager_survey_data = pd.read_csv('manager_survey_data.csv')
```


```python
data = pd.merge(general_data, manager_survey_data, on = 'EmployeeID')
data = pd.merge(data, employee_survey_data, on = 'EmployeeID')
data = data.dropna()  # Simply dropping nan
data = data.drop(['EmployeeID','EmployeeCount'],axis = 1).reset_index(drop=True)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>Gender</th>
      <th>JobLevel</th>
      <th>JobRole</th>
      <th>...</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>YearsAtCompany</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>JobInvolvement</th>
      <th>PerformanceRating</th>
      <th>EnvironmentSatisfaction</th>
      <th>JobSatisfaction</th>
      <th>WorkLifeBalance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>6</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>1</td>
      <td>Healthcare Representative</td>
      <td>...</td>
      <td>1.0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>Yes</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>10</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>...</td>
      <td>6.0</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>17</td>
      <td>4</td>
      <td>Other</td>
      <td>Male</td>
      <td>4</td>
      <td>Sales Executive</td>
      <td>...</td>
      <td>5.0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>No</td>
      <td>Non-Travel</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>5</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>3</td>
      <td>Human Resources</td>
      <td>...</td>
      <td>13.0</td>
      <td>5</td>
      <td>8</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>10</td>
      <td>1</td>
      <td>Medical</td>
      <td>Male</td>
      <td>1</td>
      <td>Sales Executive</td>
      <td>...</td>
      <td>9.0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4295</th>
      <td>29</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>4</td>
      <td>3</td>
      <td>Other</td>
      <td>Female</td>
      <td>2</td>
      <td>Human Resources</td>
      <td>...</td>
      <td>6.0</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4296</th>
      <td>42</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>4</td>
      <td>Medical</td>
      <td>Female</td>
      <td>1</td>
      <td>Research Scientist</td>
      <td>...</td>
      <td>10.0</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4297</th>
      <td>29</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>4</td>
      <td>Medical</td>
      <td>Male</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>...</td>
      <td>10.0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4298</th>
      <td>25</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>25</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>2</td>
      <td>Sales Executive</td>
      <td>...</td>
      <td>5.0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4299</th>
      <td>42</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>18</td>
      <td>2</td>
      <td>Medical</td>
      <td>Male</td>
      <td>1</td>
      <td>Laboratory Technician</td>
      <td>...</td>
      <td>10.0</td>
      <td>2</td>
      <td>9</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>4300 rows × 27 columns</p>
</div>




```python
data_hot = pd.get_dummies(data, columns = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18'])
data_hot.head() # one-hot-encoding for those with Dtype = 'object'
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>JobLevel</th>
      <th>MonthlyIncome</th>
      <th>NumCompaniesWorked</th>
      <th>PercentSalaryHike</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>...</th>
      <th>JobRole_Manager</th>
      <th>JobRole_Manufacturing Director</th>
      <th>JobRole_Research Director</th>
      <th>JobRole_Research Scientist</th>
      <th>JobRole_Sales Executive</th>
      <th>JobRole_Sales Representative</th>
      <th>MaritalStatus_Divorced</th>
      <th>MaritalStatus_Married</th>
      <th>MaritalStatus_Single</th>
      <th>Over18_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>51</td>
      <td>No</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>131160</td>
      <td>1.0</td>
      <td>11</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31</td>
      <td>Yes</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>41890</td>
      <td>0.0</td>
      <td>23</td>
      <td>8</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>No</td>
      <td>17</td>
      <td>4</td>
      <td>4</td>
      <td>193280</td>
      <td>1.0</td>
      <td>15</td>
      <td>8</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>No</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>83210</td>
      <td>3.0</td>
      <td>11</td>
      <td>8</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>No</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>23420</td>
      <td>4.0</td>
      <td>12</td>
      <td>8</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>



### 3. Defining functions 

In this part, we define some functions that can be used later on. Make sure to run these blocks before moving on.


```python
def deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k=5,
                   show_classification_report = False, show_confusion_matrix = False,
                   plot_accuracy = False, plot_recall = False, plot_f1 = False):
    
    origin = pool_classifiers           # origin means not to use any technique in deslib 
    knora_e = KNORAE(pool_classifiers, k=k).fit(x_train, y_train.values.ravel())   # Transforming the pool_classifiers into DES estimators
    desp = DESP(pool_classifiers, k=k).fit(x_train, y_train.values.ravel())
    ola = OLA(pool_classifiers, k=k).fit(x_train, y_train.values.ravel())
    rank = Rank(pool_classifiers, k=k).fit(x_train, y_train.values.ravel())
    meta = METADES(pool_classifiers, k=k).fit(x_train, y_train.values.ravel())

    y_test_predict_origin = origin.predict(x_test)
    y_test_predict_knora_e = knora_e.predict(x_test)
    y_test_predict_desp = desp.predict(x_test)
    y_test_predict_ola = ola.predict(x_test)
    y_test_predict_rank = rank.predict(x_test)
    y_test_predict_meta = meta.predict(x_test)

    pred_list = [y_test_predict_origin, y_test_predict_knora_e, y_test_predict_desp, y_test_predict_ola, y_test_predict_rank, y_test_predict_meta]
    pred_list_name = ['ORIGIN', 'KNORAE', 'DESP', 'OLA ', 'Rank ', 'METADES']
    accuracy_dic = {}; recall_dic = {}; f1_dic = {}
    accuracy_list = []; recall_list = []; f1_list = []


    for index in range(len(pred_list)):
        accuracy_dic[pred_list_name[index]] = np.round(accuracy_score(y_test.reset_index(drop=True).values, pred_list[index]), 3)
        recall_dic[pred_list_name[index]] = np.round(recall_score(y_test.reset_index(drop=True).values, pred_list[index],  pos_label = 'Yes'), 3)
        f1_dic[pred_list_name[index]] = np.round(f1_score(y_test.reset_index(drop=True).values, pred_list[index],  pos_label = 'Yes'), 3)
        
        accuracy_list.append(np.round(accuracy_score(y_test.reset_index(drop=True).values, pred_list[index]), 3))
        recall_list.append(np.round(recall_score(y_test.reset_index(drop=True).values, pred_list[index],  pos_label = 'Yes'), 3))
        f1_list.append(np.round(f1_score(y_test.reset_index(drop=True).values, pred_list[index],  pos_label = 'Yes'), 3))
        
        
    
    if show_confusion_matrix == True:
        for index in range(len(pred_list)):
            print('The confusion matrix of {}: '.format(pred_list_name[index]), end = "")
            ConfusionMatrixDisplay.from_predictions(y_test.reset_index(drop=True),pred_list[index])
            plt.show()
            
    if show_classification_report == True:
        for index in range(len(pred_list)):
            print('The classification_report of {}: '.format(pred_list_name[index]))
            print(classification_report(y_test.reset_index(drop=True), pred_list[index]))
    
    ###################### plot ######################################
    cmap = get_cmap('Dark2')
    n = len(pred_list)
    colors = [cmap(i) for i in np.linspace(0, 1, (n-1))]
    labels = pred_list_name
    
    if plot_accuracy == True:
        fig, ax = plt.subplots()
        pct_formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 100))
        ax.bar(np.arange((n)), accuracy_list, color=colors,tick_label=labels)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Method', fontsize=13)
        ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
        ax.yaxis.set_major_formatter(pct_formatter)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
    if plot_recall == True:
        fig, ax = plt.subplots()
        pct_formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 100))
        ax.bar(np.arange((n)), recall_list, color=colors,tick_label=labels)
        ax.set_ylim(0.0, 1)
        ax.set_xlabel('Method', fontsize=13)
        ax.set_ylabel('Recall on the test set (%)', fontsize=13)
        ax.yaxis.set_major_formatter(pct_formatter)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
    if plot_f1 == True:
        fig, ax = plt.subplots()
        pct_formatter = FuncFormatter(lambda x, pos: '{:.1f}'.format(x * 100))
        ax.bar(np.arange((n)), f1_list, color=colors,tick_label=labels)
        ax.set_ylim(0.0, 1)
        ax.set_xlabel('Method', fontsize=13)
        ax.set_ylabel('F1 on the test set (%)', fontsize=13)
        ax.yaxis.set_major_formatter(pct_formatter)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        

    return accuracy_dic, recall_dic, f1_dic
```


```python
def print_accuracy(accuracy_dic):
    for key, value in accuracy_dic.items():
        print(f'The accuracy performance of {key}: {value} ')
        
def print_recall(recall_dic):
    for key, value in recall_dic.items():
        print(f'The recall performance of {key}: {value} ')

def print_f1(f1_dic):
    for key, value in f1_dic.items():
        print(f'The f1 performance of {key}: {value} ')

```

### 4. Start to train - Deslib - AdaBoost

In this part, the base ensemble method is Adaptive Boost, and the function deslib_process returns all the performance of DES models based on AdaBoost. Set True for the plots you want to see. For example, if you want to see the plot ot accuracy comparasion, set plot_accuracy = True.

"AdaBoostClassifier" comes from sklearn.


```python
y = data_hot[['Attrition']]
x = data_hot.drop(['Attrition'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7404)
```


```python
pool_classifiers = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),  
                                      n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = True, plot_recall = False, plot_f1 = False)
```


    
![png](output_16_0.png)
    



```python
print_accuracy(accuracy_dic)  # You can extract the values and names in the format of dictionary
```

    The accuracy performance of ORIGIN: 0.985 
    The accuracy performance of KNORAE: 0.988 
    The accuracy performance of DESP: 0.912 
    The accuracy performance of OLA : 0.987 
    The accuracy performance of Rank : 0.987 
    The accuracy performance of METADES: 0.995 
    


```python
pool_classifiers = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),  # max_depth selection is very important for deslib's enhencement
                                      n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = True, plot_f1 = False)
```


    
![png](output_18_0.png)
    



```python
print_recall(recall_dic) 
```

    The recall performance of ORIGIN: 0.939 
    The recall performance of KNORAE: 0.939 
    The recall performance of DESP: 0.523 
    The recall performance of OLA : 0.949 
    The recall performance of Rank : 0.949 
    The recall performance of METADES: 0.967 
    


```python
pool_classifiers = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),  # max_depth selection is very important for deslib's enhencement
                                      n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = False, plot_f1 = True)
```


    
![png](output_20_0.png)
    



```python
# print_f1(f1_dic)
print_f1(f1_dic) # You can extract the values and names in the format of dictionary
```

    The f1 performance of ORIGIN: 0.955 
    The f1 performance of KNORAE: 0.962 
    The f1 performance of DESP: 0.665 
    The f1 performance of OLA : 0.96 
    The f1 performance of Rank : 0.96 
    The f1 performance of METADES: 0.983 
    

### 5. Start to train - Deslib - BaggingClassifier

In this part, the base ensemble method is Bagging, and the function deslib_process returns all the performance of DES models based on Bagging Classifier. 

"BaggingClassifier" comes from sklearn.


```python
pool_classifiers = BaggingClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = True, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = False, plot_f1 = False)
```

    The classification_report of ORIGIN: 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       1.00      0.95      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    The classification_report of KNORAE: 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       1.00      0.95      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    The classification_report of DESP: 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       1.00      0.95      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    The classification_report of OLA : 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       0.99      0.96      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.98      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    The classification_report of Rank : 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       0.99      0.96      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.98      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    The classification_report of METADES: 
                  precision    recall  f1-score   support
    
              No       0.99      1.00      0.99      1076
             Yes       1.00      0.95      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    


```python
pool_classifiers = BaggingClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = True, plot_recall = False, plot_f1 = False)
```


    
![png](output_24_0.png)
    



```python
print_accuracy(accuracy_dic)
```

    The accuracy performance of ORIGIN: 0.991 
    The accuracy performance of KNORAE: 0.991 
    The accuracy performance of DESP: 0.991 
    The accuracy performance of OLA : 0.991 
    The accuracy performance of Rank : 0.991 
    The accuracy performance of METADES: 0.991 
    


```python
pool_classifiers = BaggingClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = True, plot_f1 = False)
```


    
![png](output_26_0.png)
    



```python
print_recall(recall_dic) 
```

    The recall performance of ORIGIN: 0.949 
    The recall performance of KNORAE: 0.949 
    The recall performance of DESP: 0.949 
    The recall performance of OLA : 0.963 
    The recall performance of Rank : 0.963 
    The recall performance of METADES: 0.949 
    


```python
pool_classifiers = BaggingClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = False, plot_f1 = True)
```


    
![png](output_28_0.png)
    



```python
print_f1(f1_dic)
```

    The f1 performance of ORIGIN: 0.974 
    The f1 performance of KNORAE: 0.974 
    The f1 performance of DESP: 0.974 
    The f1 performance of OLA : 0.974 
    The f1 performance of Rank : 0.974 
    The f1 performance of METADES: 0.974 
    

### 6. Start to train - Random Forest

In this part, the base ensemble method is Random Forest, and the function deslib_process returns all the performance of DES models based on Random Forest Classifier. 

"RandomForestClassifier" comes from sklearn.


```python
pool_classifiers = RandomForestClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = True, plot_recall = False, plot_f1 = False)
```


    
![png](output_31_0.png)
    



```python
print_accuracy(accuracy_dic)
```

    The accuracy performance of ORIGIN: 0.991 
    The accuracy performance of KNORAE: 0.991 
    The accuracy performance of DESP: 0.991 
    The accuracy performance of OLA : 0.983 
    The accuracy performance of Rank : 0.983 
    The accuracy performance of METADES: 0.991 
    


```python
pool_classifiers = RandomForestClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = True, plot_f1 = False)
```


    
![png](output_33_0.png)
    



```python
print_recall(recall_dic) 
```

    The recall performance of ORIGIN: 0.949 
    The recall performance of KNORAE: 0.949 
    The recall performance of DESP: 0.949 
    The recall performance of OLA : 0.981 
    The recall performance of Rank : 0.981 
    The recall performance of METADES: 0.949 
    


```python
pool_classifiers = RandomForestClassifier(n_estimators=100, random_state=7404)
pool_classifiers.fit(x_train, y_train.values.ravel())
accuracy_dic, recall_dic, f1_dic = deslib_process(pool_classifiers, x_train, x_test, y_train, y_test, k = 6,
                                            show_classification_report = False, show_confusion_matrix = False,
                                            plot_accuracy = False, plot_recall = False, plot_f1 = True)
```


    
![png](output_35_0.png)
    



```python
print_f1(f1_dic)
```

    The f1 performance of ORIGIN: 0.974 
    The f1 performance of KNORAE: 0.974 
    The f1 performance of DESP: 0.974 
    The f1 performance of OLA : 0.95 
    The f1 performance of Rank : 0.95 
    The f1 performance of METADES: 0.974 
    

### 7. Start to train - Neural Network

Run the code subsequently and everything should be out of trouble.

Notice that we re-define the input data here so as to fit the format of the Neural Network. The output layer only consists of one node, generated by sigmoid function, so we only extract the value Attrition == Yes or No (map to 1 or 0).

Also notice that the result may be different everytime, because there is randomness in Neural Network.(IMPORTANT, might be different with the result shown in our presentation.)


```python
y = data_hot[['Attrition']]
x = data_hot.drop(['Attrition'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7404)
```


```python
scaler = MinMaxScaler(feature_range=(0,1))     # Min Max Scaling is very important in DNN
x_train = pd.DataFrame(scaler.fit_transform(np.array(x_train)))
x_test = pd.DataFrame(scaler.fit_transform(x_test))
```


```python
y_train = pd.get_dummies(y_train)[['Attrition_Yes']]
y_test = pd.get_dummies(y_test)[['Attrition_Yes']]   # get dummy, only one output
```


```python
input_shape = x_train.shape[1] # 46
```


```python
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (input_shape,)))
network.add(layers.Dense(256, activation = 'relu'))
network.add(layers.Dense(128, activation = 'relu'))
network.add(layers.Dense(1, activation = 'sigmoid')) # output layer
network.compile(optimizer = 'adam', loss = losses.binary_crossentropy, metrics = ['accuracy'])
```


```python
network.fit(x_train, y_train, epochs = 100, batch_size = 128)
```

    Epoch 1/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.4535 - accuracy: 0.8169
    Epoch 2/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.3900 - accuracy: 0.8402
    Epoch 3/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.3571 - accuracy: 0.8545
    Epoch 4/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.3203 - accuracy: 0.8754
    Epoch 5/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.2736 - accuracy: 0.8887
    Epoch 6/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.2179 - accuracy: 0.9209
    Epoch 7/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.1739 - accuracy: 0.9282
    Epoch 8/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.1246 - accuracy: 0.9608
    Epoch 9/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0884 - accuracy: 0.9748
    Epoch 10/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0605 - accuracy: 0.9834
    Epoch 11/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0612 - accuracy: 0.9807
    Epoch 12/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0564 - accuracy: 0.9801
    Epoch 13/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0267 - accuracy: 0.9944
    Epoch 14/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0116 - accuracy: 0.9987
    Epoch 15/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0065 - accuracy: 0.9990
    Epoch 16/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0065 - accuracy: 0.9990
    Epoch 17/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.0040 - accuracy: 0.9993
    Epoch 18/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0037 - accuracy: 0.9993
    Epoch 19/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0069 - accuracy: 0.9980
    Epoch 20/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0126 - accuracy: 0.9983
    Epoch 21/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.0048 - accuracy: 0.9990
    Epoch 22/100
    24/24 [==============================] - 0s 2ms/step - loss: 0.0016 - accuracy: 1.0000
    Epoch 23/100
    24/24 [==============================] - 0s 3ms/step - loss: 0.0010 - accuracy: 1.0000
    Epoch 24/100
    24/24 [==============================] - 0s 3ms/step - loss: 6.2506e-04 - accuracy: 1.0000
    Epoch 25/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.2419e-04 - accuracy: 1.0000
    Epoch 26/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.4139e-04 - accuracy: 1.0000
    Epoch 27/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.9050e-04 - accuracy: 1.0000
    Epoch 28/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.5162e-04 - accuracy: 1.0000
    Epoch 29/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.1449e-04 - accuracy: 1.0000
    Epoch 30/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.8825e-04 - accuracy: 1.0000
    Epoch 31/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.6456e-04 - accuracy: 1.0000
    Epoch 32/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.3981e-04 - accuracy: 1.0000
    Epoch 33/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.2177e-04 - accuracy: 1.0000
    Epoch 34/100
    24/24 [==============================] - 0s 3ms/step - loss: 2.0497e-04 - accuracy: 1.0000
    Epoch 35/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.9114e-04 - accuracy: 1.0000
    Epoch 36/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.7696e-04 - accuracy: 1.0000
    Epoch 37/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.6540e-04 - accuracy: 1.0000
    Epoch 38/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.5439e-04 - accuracy: 1.0000
    Epoch 39/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.4397e-04 - accuracy: 1.0000
    Epoch 40/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.3621e-04 - accuracy: 1.0000
    Epoch 41/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.2792e-04 - accuracy: 1.0000
    Epoch 42/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.1530e-04 - accuracy: 1.0000
    Epoch 43/100
    24/24 [==============================] - 0s 2ms/step - loss: 9.9592e-05 - accuracy: 1.0000
    Epoch 44/100
    24/24 [==============================] - 0s 2ms/step - loss: 8.9087e-05 - accuracy: 1.0000
    Epoch 45/100
    24/24 [==============================] - 0s 2ms/step - loss: 7.9295e-05 - accuracy: 1.0000
    Epoch 46/100
    24/24 [==============================] - 0s 2ms/step - loss: 7.2545e-05 - accuracy: 1.0000
    Epoch 47/100
    24/24 [==============================] - 0s 2ms/step - loss: 6.3745e-05 - accuracy: 1.0000
    Epoch 48/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.6183e-05 - accuracy: 1.0000
    Epoch 49/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.0167e-05 - accuracy: 1.0000
    Epoch 50/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.5205e-05 - accuracy: 1.0000
    Epoch 51/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.1083e-05 - accuracy: 1.0000
    Epoch 52/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.7850e-05 - accuracy: 1.0000
    Epoch 53/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.4360e-05 - accuracy: 1.0000
    Epoch 54/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.0935e-05 - accuracy: 1.0000
    Epoch 55/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.8084e-05 - accuracy: 1.0000
    Epoch 56/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.5659e-05 - accuracy: 1.0000
    Epoch 57/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.3720e-05 - accuracy: 1.0000
    Epoch 58/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.2056e-05 - accuracy: 1.0000
    Epoch 59/100
    24/24 [==============================] - 0s 2ms/step - loss: 2.0042e-05 - accuracy: 1.0000
    Epoch 60/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.8515e-05 - accuracy: 1.0000
    Epoch 61/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.7335e-05 - accuracy: 1.0000
    Epoch 62/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.6249e-05 - accuracy: 1.0000
    Epoch 63/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.5205e-05 - accuracy: 1.0000
    Epoch 64/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.4241e-05 - accuracy: 1.0000
    Epoch 65/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.3279e-05 - accuracy: 1.0000
    Epoch 66/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.2518e-05 - accuracy: 1.0000
    Epoch 67/100
    24/24 [==============================] - 0s 2ms/step - loss: 1.2013e-05 - accuracy: 1.0000
    Epoch 68/100
    24/24 [==============================] - 0s 3ms/step - loss: 1.1246e-05 - accuracy: 1.0000
    Epoch 69/100
    24/24 [==============================] - 0s 3ms/step - loss: 1.0575e-05 - accuracy: 1.0000
    Epoch 70/100
    24/24 [==============================] - 0s 3ms/step - loss: 1.0017e-05 - accuracy: 1.0000
    Epoch 71/100
    24/24 [==============================] - 0s 2ms/step - loss: 9.5647e-06 - accuracy: 1.0000
    Epoch 72/100
    24/24 [==============================] - 0s 2ms/step - loss: 9.0528e-06 - accuracy: 1.0000
    Epoch 73/100
    24/24 [==============================] - 0s 2ms/step - loss: 8.5939e-06 - accuracy: 1.0000
    Epoch 74/100
    24/24 [==============================] - 0s 2ms/step - loss: 8.2195e-06 - accuracy: 1.0000
    Epoch 75/100
    24/24 [==============================] - 0s 2ms/step - loss: 7.7951e-06 - accuracy: 1.0000
    Epoch 76/100
    24/24 [==============================] - 0s 2ms/step - loss: 7.4482e-06 - accuracy: 1.0000
    Epoch 77/100
    24/24 [==============================] - 0s 2ms/step - loss: 7.1485e-06 - accuracy: 1.0000
    Epoch 78/100
    24/24 [==============================] - 0s 2ms/step - loss: 6.8589e-06 - accuracy: 1.0000
    Epoch 79/100
    24/24 [==============================] - 0s 2ms/step - loss: 6.5271e-06 - accuracy: 1.0000
    Epoch 80/100
    24/24 [==============================] - 0s 2ms/step - loss: 6.2445e-06 - accuracy: 1.0000
    Epoch 81/100
    24/24 [==============================] - 0s 2ms/step - loss: 6.0181e-06 - accuracy: 1.0000
    Epoch 82/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.8244e-06 - accuracy: 1.0000
    Epoch 83/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.5569e-06 - accuracy: 1.0000
    Epoch 84/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.3512e-06 - accuracy: 1.0000
    Epoch 85/100
    24/24 [==============================] - 0s 2ms/step - loss: 5.1361e-06 - accuracy: 1.0000
    Epoch 86/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.9502e-06 - accuracy: 1.0000
    Epoch 87/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.7664e-06 - accuracy: 1.0000
    Epoch 88/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.6003e-06 - accuracy: 1.0000
    Epoch 89/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.4620e-06 - accuracy: 1.0000
    Epoch 90/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.3132e-06 - accuracy: 1.0000
    Epoch 91/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.1391e-06 - accuracy: 1.0000
    Epoch 92/100
    24/24 [==============================] - 0s 2ms/step - loss: 4.0315e-06 - accuracy: 1.0000
    Epoch 93/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.8612e-06 - accuracy: 1.0000
    Epoch 94/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.7544e-06 - accuracy: 1.0000
    Epoch 95/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.6341e-06 - accuracy: 1.0000
    Epoch 96/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.5052e-06 - accuracy: 1.0000
    Epoch 97/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.3908e-06 - accuracy: 1.0000
    Epoch 98/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.2948e-06 - accuracy: 1.0000
    Epoch 99/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.1904e-06 - accuracy: 1.0000
    Epoch 100/100
    24/24 [==============================] - 0s 2ms/step - loss: 3.0936e-06 - accuracy: 1.0000
    




    <keras.callbacks.History at 0x22d0cceb5e0>




```python
test_loss, test_acc = network.evaluate(x_test, y_test)
```

    41/41 [==============================] - 0s 1ms/step - loss: 0.0680 - accuracy: 0.9930
    


```python
pred_y_test = pd.DataFrame(np.around(network.predict(x_test), 1).astype(int)).reset_index(drop=True)
real_y_test = y_test.reset_index(drop=True)

print(classification_report(real_y_test, pred_y_test))
# confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(real_y_test, pred_y_test)
plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99      1076
               1       1.00      0.95      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    


    
![png](output_45_1.png)
    



```python
print(f'The accuracy performance of DNN: {np.round(accuracy_score(real_y_test, pred_y_test), 3)} ')
print(f'The recall performance of DNN: {np.round(recall_score(real_y_test, pred_y_test), 3)} ')
print(f'The f1 performance of DNN: {np.round(f1_score(real_y_test, pred_y_test), 3)} ')
```

    The accuracy performance of DNN: 0.993 
    The recall performance of DNN: 0.958 
    The f1 performance of DNN: 0.979 
    

### 8. SVM

Run the code subsequently and everything should be out of trouble.

The block of grid_search might take some time but don't worry.


```python
y = data_hot[['Attrition']]
x = data_hot.drop(['Attrition'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7404)
```


```python
scaler = MinMaxScaler(feature_range=(0,1))
x_train = pd.DataFrame(scaler.fit_transform(np.array(x_train)))
x_test = pd.DataFrame(scaler.fit_transform(x_test))
```


```python
y_train = pd.get_dummies(y_train)[['Attrition_Yes']]
y_test = pd.get_dummies(y_test)[['Attrition_Yes']]   # get dummy, only one output
```


```python
tuned_parameters = [{'C': np.logspace(-3, 3, 7),
                     'gamma': np.logspace(-3, 3, 7)}]
model = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=5)
model.fit(x_train, y_train.values.ravel())
model.best_params_
```




    {'C': 1.0, 'gamma': 10.0}




```python
pred_y_test = pd.DataFrame(model.predict(x_test)).reset_index(drop=True)
real_y_test = y_test.reset_index(drop=True)
```


```python
print(f'The accuracy performance of DNN: {np.round(accuracy_score(real_y_test, pred_y_test), 3)} ')
print(f'The recall performance of DNN: {np.round(recall_score(real_y_test, pred_y_test), 3)} ')
print(f'The f1 performance of DNN: {np.round(f1_score(real_y_test, pred_y_test), 3)} ')
```

    The accuracy performance of DNN: 0.99 
    The recall performance of DNN: 0.939 
    The f1 performance of DNN: 0.969 
    


```python
print(classification_report(real_y_test, pred_y_test))
# confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(real_y_test, pred_y_test)
plt.show()
```

                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99      1076
               1       1.00      0.94      0.97       214
    
        accuracy                           0.99      1290
       macro avg       0.99      0.97      0.98      1290
    weighted avg       0.99      0.99      0.99      1290
    
    


    
![png](output_54_1.png)
    


### Explore Data Analysis (EDA)

We didn't include the HR_EDA.ipynb into Demo video for two reasons:
1. We don't have time to include it since the Demo video should be less than 2 minutes.
2. It doesn't belong to the models building part, so it doesn't affect the models even if you cannot run this code successfully.
3. Actually, just run the blocks subsequently and everything should be fine.

In this README file, we still provide some instructions.

### 1. Installing Python packages

Please make sure you have installed the following packages successfully. The version of sklearn should also be updated.
Then run every block to import the packages.


```python
!pip install chart_studio
!pip install deslib
!pip install sklearn
!pip install keras
!pip install tensorflow
!pip install matplotlib
```

    Collecting chart_studio
      Downloading chart_studio-1.1.0-py3-none-any.whl (64 kB)
    Collecting retrying>=1.3.3
      Downloading retrying-1.3.3.tar.gz (10 kB)
    Requirement already satisfied: plotly in d:\anaconda\lib\site-packages (from chart_studio) (5.6.0)
    Requirement already satisfied: requests in d:\anaconda\lib\site-packages (from chart_studio) (2.27.1)
    Requirement already satisfied: six in d:\anaconda\lib\site-packages (from chart_studio) (1.16.0)
    Requirement already satisfied: tenacity>=6.2.0 in d:\anaconda\lib\site-packages (from plotly->chart_studio) (8.0.1)
    Requirement already satisfied: charset-normalizer~=2.0.0 in d:\anaconda\lib\site-packages (from requests->chart_studio) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in d:\anaconda\lib\site-packages (from requests->chart_studio) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in d:\anaconda\lib\site-packages (from requests->chart_studio) (2022.6.15)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in d:\anaconda\lib\site-packages (from requests->chart_studio) (1.26.9)
    Building wheels for collected packages: retrying
      Building wheel for retrying (setup.py): started
      Building wheel for retrying (setup.py): finished with status 'done'
      Created wheel for retrying: filename=retrying-1.3.3-py3-none-any.whl size=11447 sha256=6b67304c49880f766ba55a1eff086fcdbb357d98aacc52d1c7da7ea9c6b66061
      Stored in directory: c:\users\zeng's yoga\appdata\local\pip\cache\wheels\ce\18\7f\e9527e3e66db1456194ac7f61eb3211068c409edceecff2d31
    Successfully built retrying
    Installing collected packages: retrying, chart-studio
    Successfully installed chart-studio-1.1.0 retrying-1.3.3
    

### 2. Run the blocks subsequently

As mentioned before, Please make sure that all of these 3 csv files are PLACED IN THE SAME DIRECTORY of the HR_EDA.ipynb's. 

Nothing special, you only need to run every block subsequently in jupyter notebook, there should not be any trouble.


```python

```


```python

```


```python

```


```python

```
