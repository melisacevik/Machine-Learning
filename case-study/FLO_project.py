###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik
# olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################
# Görev 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okutunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
           # Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# Görev 2: K-Means ile Müşteri Segmentasyonu
           # Değişkenleri standartlaştırınız.
           # Optimum küme sayısını belirleyiniz.
           # Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
           # Herbir segmenti istatistiksel olarak inceleyiniz.


# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
           # Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
           # Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
           # Her bir segmenti istatistiksel olarak inceleyiniz.

## Kütüphaneler

import datetime as dt
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


# 1. flo_data_20k.csv verisini okutunuz.

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy() 

df.head()

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

# 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.

# Tarih ifade eden değişkenlerin tipini date'e çevirme

df.head()
date_list = ["first_order_date", "last_order_date","last_order_date_online", "last_order_date_offline"]

for column in date_list:
    df[column] = pd.to_datetime(df[column], format="%Y-%m-%d")

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)

df['recency'] = (df['last_order_date'] - df['first_order_date']).dt.days

# müşterinin offline ve online'da yaptığı alışveriş sayısı - frequency

df["frequency"] = (df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]).astype(int)

# müşterinin alışverişlerindeki toplam harcama
df["value_count"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# müşterinin yaşı

df['customer_age'] = (today_date - df['first_order_date']).dt.days

df['monetary'] = df['value_count'] / df['frequency']

df.head()

###############################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
###############################################################

######
# Adım 1: Değişkenleri standartlaştırınız.
######


# veri yapısı Object olan feature'larımı one-hot-encoder işleminden geçiriyorum.

# master_id'yi ve tarih değişkenlerimi kaldırıyorum. ( zaten recency vs. oluşturdum)

drop_list = ["master_id", "first_order_date", "last_order_date","last_order_date_online", "last_order_date_offline"]

df.drop(drop_list, axis=1, inplace=True)

df.head()


# object veri tipinde olanlar : order_channel(4) , last_order_channel(5) , interested_in_categories_12(32)

# one-hot encoding işleminden geçiriyorum.
for column in df.columns:
    if df[column].dtype == 'object' and df[column].nunique() >= 2:
        df = pd.get_dummies(df, columns=[column], dtype=int)

df.info()

# artık bütün değişkenler int,float tipinde oldu.

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

######
# 2. Optimum küme sayısını belirleyiniz.
######

kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans= KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K,ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

######
# 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz
######

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters #10
kmeans.cluster_centers_ #10 array
kmeans.labels_

clusters_kmeans = kmeans.labels_

# 4. Her bir segmenti istatistiksel olarak inceleyeniz.

df = df_.copy()
df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1

df[df["cluster"]==5]

# istatistiksel olarak incelemek için objectlerden kurtuluyorum.

df.info()

drop_obj_columns = ["master_id", "order_channel", "last_order_channel", "first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline", "interested_in_categories_12"]

df.drop(drop_obj_columns, axis=1, inplace=True)

df.head()

df.groupby("cluster").agg(["count","mean","median"])

######
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
######

# 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

# Görev1. ve 4. adımın dışında tekrar çalıştır.

df[0:5]

hc_average = linkage(df, "average")

# hiyerarşik bir kümeleme dendrogramı

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
          leaf_font_size = 10)
plt.show()

# küme sayısını belirleme

plt.figure(figsize=(7,5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,truncate_mode="lastp",
           p=10,
           show_contracted=True)
plt.axhline(y=1, color='r', linestyle="--")
plt.show()

# küme sayısı : 10 

from sklearn.cluster import AgglomerativeClustering

hi_cluster = AgglomerativeClustering(n_clusters = 5, linkage="average")

hi_clusters = hi_cluster.fit_predict(df)

df = df_.copy()
df["hi_cluster_no"] = hi_clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1
df.head()

# kmeans'leri de getir.

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df.head()


# 3. Her bir segmenti istatistiksel olarak inceleyeniz

drop_obj_columns = ["master_id", "order_channel", "last_order_channel", "first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline", "interested_in_categories_12"]
df.drop(drop_obj_columns, axis=1, inplace=True)
df.groupby("hi_cluster_no").agg(["count","mean","median"])

