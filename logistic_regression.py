######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,plot_roc_curve #sonuncusu import edilmedi
from sklearn.model_selection import train_test_split, cross_validate
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None) # bütün sütunları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x) # virgülden sonra 3 rakam al
pd.set_option('display.width', 500) # mümkün old.kadar geniş



######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df) # 1 ve 0 'ın oranı

##########################
# Feature'ların Analizi
##########################

df.describe().T # sadece sayısal değişkenleri özetler
df.head()

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]


# for col in cols:
#     plot_numerical_col(df, col)

df.describe().T

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"}) # hamilelik sayısı değişkeni için diyabet hastası olanların ort

# target'i numerik değişkenle özetle
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)



######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()

df.isnull().sum()

df.describe().T

# thresold belirle ( check içinde var ) ve var mı bak
for col in cols:
    print(col, check_outlier(df, col))

#thresholdu yerine aykrııların yerine yaz.
replace_with_thresholds(df, "Insulin")

# değişkenleri scale et / ölçeklendir
#robust => bütün gözlem biriminin değerlerinden medyanı çıkarıp range değerine bölüyor.
# standart scaler dan farkı daha robust yani aykırı değerlerden etkilenmiyor olması
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################
#amaç bağımlı ve bağımsız değişkenler arasındaki ilişkiyi modellemek
# lojistik regresyon yöntemini kullanıyorduk.

#bağımlı ve bağımsız değişkenleri ayrı ayrı ifade etmemiz lazım.
y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_ #sabit
log_model.coef_ #bağımsız değişkenkerin ağırlıkları

y_pred = log_model.predict(X)  #tahmin işlemi

y_pred[0:10]

y[0:10]

# modeli kurduk
# ne kadar başarılı ?

######################################################
# Model Evaluation
######################################################

# recall, precision , f1, accuracy => karmaşıklık matrixindeki değerleri numerikleştirmek için => confusion_matrix() metodu kullanırız.
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# iki alt sütunlar gelmedi - buraya dikkat


# Accuracy: 0.78 : doğru sınıflandırma oranı : 156 + 112 / hepsi
# Precision: 0.74 : 1 olarak yaptığımız tahminler ne kadar başarılı : 156 / ( 156 + 54 )
# Recall: 0.58 : 1 olanları ne kadar başarılı tahmin etmişiz : 112 + ( 112 + 156 )
# F1-score: 0.65 : pre + recall 'un harmonik ortalaması


# ROC AUC : farklı classification th değerlerine göre oluşabilecek başarılarımıza yönelik genel bir metrik
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939


# veri setindeki verilerle test etmiş oldum  bunun için holdout yöntemine bak

######################################################
# Model Validation: Holdout
######################################################

# verisetini train ve test olarak 80 20 olacak şekilde ikiye ayır.

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# random _ state ile aynı test veri setini oluşturmuş olurum

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test) #eğitim yaparken gösterilmemiş veri , o verinin bağımsız değişkenlerine göre diyabet olup olmama durumunda tahminde bulun

# tahmin edilen değerlerle beraber, bir de 1 sınıfına ait olma olasılıklarını da hesapla
y_prob = log_model.predict_proba(X_test)[:, 1]

# başarımızı değerlendirelim
print(classification_report(y_test, y_pred)) #y_pred => az önce tahmin ettiğimiz değerler | y_test => test setindeki gerçek y değerleri(modelin görmedi yer)

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# model başarısı ile ilgili bir grafik oluşturuyor.

plot_roc_curve(log_model, X_test, y_test)
#plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


# AUC
roc_auc_score(y_test, y_prob)

# daha önceki 0.84 tü şimdi 0.88 oldu.
# random_state'i farklı aldığımızda farklı sonuç geliyor.

# bunu önlemek için 10-fold cross validation kullanırız.



######################################################
# Model Validation: 10-Fold Cross Validation
######################################################


# modelin doğrulama sürecini en doğru şekilde ele alalım. bunu yapmanın yolu da çapraz doğrulama işlemidir.
# model veri setinin farklı parçalarıyla modellenip farklı parçalarıyla test edilmiş olacağından olası tüm senaryolara yönelik bir fikir veriyor olur.



y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# burada tüm veriyi kullanarak çapraz doğrulama işlemini gerçekleştiricez ( az diye )

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)

