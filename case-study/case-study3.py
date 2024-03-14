##########################################################
# İş Problemi : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
##########################################################

# Veri Seti Hikayesi : Telco müşteri kaybı verileri,
# üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import missingno as msno
from pandas.core.interchange import dataframe
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

dff = pd.read_csv('datasets/Telco-Customer-Churn.csv')
df = dff.copy()
df.head()

# Görev 1: Keşifçi Veri Analizi (EDA)

def check_df(dataframe, head=5):
    print("######################## Shape ############################")
    print(dataframe.shape)
    print("######################## Types ############################")
    print(dataframe.dtypes)
    print("######################## Head #############################")
    print(dataframe.head(head))
    print("######################## Tail #############################")
    print(dataframe.tail(head))
    print("######################## NA ###############################")
    print(dataframe.isnull().sum())
    print("####################### Quantiles #########################")
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

# totalCharges => veri tipi değişecek float olacak

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi) ( TotalCharges )

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    dataframe['TotalCharges'] = pd.to_numeric(dataframe['TotalCharges'], errors='coerce')
    num_cols.append('TotalCharges')
    cat_but_car = [col for col in cat_but_car if col not in "TotalCharges"] # bu numerik değişken olmamalı Id var

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

# numerik değişkenlerin dağılımı
for col in num_cols:
    df.boxplot(column=col)
    plt.title(col)
    plt.show()

# kategorik değişkenlerin dağılımı

for col in cat_cols:
    plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%')
    plt.title(col)
    plt.show()

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.

# target : churn
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

# Yorum:

# Gender : kadınların churn oranı biraz daha yüksek. ( %0.8)
# Partner : eşi olmayan müşterilerin churn oranı daha yüksektir.
# Dependents : Bakmakla yükümlü kişiler olmayan müşterilerin churn oranı daha yüksektir.
# PhoneService : Telefon hizmeti olan müşterilerin churn oranı daha yüksektir.
# MultipleLines : Birden fazla hattı olan müşterilerin churn oranı daha yüksektir.
# InternetService : churn oranı: Fiber optik internet servisine sahip müşteriler > DSL > olmayan
# OnlineSecurity : Online güvenlik hizmetine sahip olmayan müşterilerin churn oranı daha yüksektir.
# OnlineBackup : Çevrimiçi yedekleme hizmeti almayan müşterilerin churn oranı daha yüksektir.
# DeviceProtection : Cihaz koruma hizmeti almayan müşterilerin churn oranı daha yüksektir.
# TechSupport : Teknik Desteğe sahip olmayan müşterilerin churn oranı daha yüksektir.
# StreamingTV : Akıllı TV hizmeti almayan müşterilerin churn oranı daha yüksektir.
# StreamingMovie : Akıllı film izleme hizmeti almayan müşterilerin churn oranı daha yüksektir. ( çok belirgin fark yok! )
# Contract : aylık sözleşmeye sahip müşterilerin churn oranı daha yüksektir.
# PaperlessBilling : Kağıtsız fatura tercih eden müşterilerin churn oranı daha yüksektir.
# PaymentMethod : Elektronik çek ödeme yöntemini kullanan müşterilerin churn oranı daha yüksektir.

# Adım 5: Aykırı gözlem var mı inceleyiniz.

def outlier_graph(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    plt.figure(figsize=(15, 20))
    for i, column in enumerate(num_cols):
        plt.subplot(len(num_cols) // 2 + 1, 2, i+1)
        sns.boxplot(x=df[column], color="red")
        plt.title(f"Boxplot of {column}", pad=20)

    plt.tight_layout()

outlier_graph(df)
plt.show()

# Adım 6: Eksik gözlem var mı inceleyiniz.

msno.bar(df)
plt.show()

df.head(50)
df.describe().T
# Görev 2: Feature Engineering

#Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

#Eksik değer

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df.isnull().sum()

#Aykırı değerler için de çok aykırı olanları temizleme işlemi yapacağım.

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

for col in num_cols:
    print(col ,check_outlier(df,col))
# Aykırı değer yok


# korelasyon
df.columns
corr_matrix = df[num_cols].corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()

#Adım 2: Yeni değişkenler oluşturunuz.
# Tenure değişkeninden yıllık kategorik değişken oluşturma

df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

#Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme

df["NEW_Engaged"] = df["Contract"].apply(lambda x:1 if x in ["One year","Two year"] else 0)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Streaming hizmeti alan kişiler

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

#Adım 3:  Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# 2 sınıflı olan kategorilere binary encoding işlemini yapıyorum.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols
# 3 sınıflı olan kategorileri one-hot encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Adım 1:  Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.


y = df["Churn"]
X = df.drop(["Churn",'customerID'], axis=1)

df.isnull().sum()


# model

knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user)

# model başarısı

y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))

roc_auc_score(y, y_prob)

#cross validation

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() #0.76
cv_results["test_f1"].mean() # 0.55
cv_results["test_roc_auc"].mean() # 0.78

# Adım 2: Seçtiğiniz modeller ile hiperparametreoptimizasyonu gerçekleştirin ve bulduğunuz hiparparametrelerile modeli tekrar kurunuz.

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2,50)} # 2 den 50'ye sayı oluşturduk

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1 ).fit(X, y) # n_jobs = sonuca hızlı gitmek için kullanılır. # rapor için verbose=1

knn_gs_best.best_params_

# 20 geldi.

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() #0.79
cv_results["test_f1"].mean() # 0.57
cv_results["test_roc_auc"].mean() #0.82

random_user = X.sample(1)

knn_final.predict(random_user)
