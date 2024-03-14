################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

################################################
# 2. Data Preprocessing & Feature Engineering
################################################
# bağımlı ve bağımsız değişkenleri ayır.

y = df["Outcome"]
X = df.drop("Outcome", axis = 1)

# KNN de ve Gradient Descent yönteminde değişkenlerin standart olması elde edilecek sonuçların hızlı,doğru veya başarılı olmasını sağlar!

X_scaled = StandardScaler().fit_transform(X) #standartlaştırdık, numpy array dönüyor ve istediğimiz bilgiyi taşımıyor
# değişken isimleri yok o yüzden df'e çevir.

X = pd.DataFrame(X_scaled, columns =X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y) #bağımlı ve bağımsız değişkenlerimi fit ettim.

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# model başarısı

# knn modeli kullanarak bütün gözlem birimleri için tahmin yapıp bunları bir yerde saklamamız lazım.

y_pred = knn_model.predict(X) # bütün gözlem değerleri için
# bunu confusion matrix i hesaplamak için kullanacağız.


#AUC için y_prob : 1 sınıfına ait olma olasılıklarını getirdim. Bu olasılıklar üzerinden roc auc skoru hesaplayacağız.
y_prob = knn_model.predict_proba(X)[:, 1]

# confusion matrix için tahmin edilen değerler elimizde. classification_report metodunu getirerek hesaplama işlemini gerçekleştirelim.

print(classification_report(y, y_pred))
# AYNI VERİ ÜZERİNDEN HEM MODEL KURDUK HEM TEST ETTİK. GÜVENEMEM. SONUÇLARI DOĞRULAMAK İÇİN CROSS VALIDATE YAPMALIYIZ

#AUC
roc_auc_score(y, y_prob)

# bütün veriyle model kurduk, bütün veriyle test ettik ama bu yanlış! o yüzden çapraz doğrulama yöntemiyle doğrulama yapmam gerekir.
# hata değerlendirme ( 5 katlı çapraz )

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring=["accuracy", "f1", "roc_auc"]) #model nesnen, bağımsız, bağımlı, kaç katlı olsun, kullanmak istediğin metrikleri ver
# 4 ' ü ile model kurdu 1'i ile test etti. bunların her birinin sonucu farklı
# 'test_roc_auc': array([0.77555556, 0.78759259, 0.73194444, 0.83226415, 0.77528302])}
# bunların ortalamasını aldığımızda 5 katlı çapraz doğrulamanın bütün test skorlarının ortalamasını almış olacağız.


cv_results["test_accuracy"].mean() #0.73
cv_results["test_f1"].mean() # 0.59
cv_results["test_roc_auc"].mean() # 0.78

# daha da düştü neden?
# modeli kurduğumuz veriyi modelin performansını değerlendirmek için kullandığımızda aslında ortaya bir miktar yanlılık çıktı.
# sonuçları doğru değerlendirmemizi engeller.

# çıkarmamız gereken sonuç : veri setini cross validation yöntemiyle bölerek ayrı parçalarında model kurup,
# diğer parçalarında test ettiğimizde, bütün veriyle model kurmaya göre daha farklı sonuç aldık ve bu sonuçlar daha güvenilirdir.

# bu başarı skorları nasıl artırılabilir?
# 1 - Veri boyutu artırılabilir. ( gözlem sayısını artırma )
# 2- Veri ön işleme işlemleri detaylandırılabilir.
# 3- Feature Engineering, yeni değişkenler türetilebilir.
# 4- İlgili algoritma için optimizasyon yapılabilir.



#KNN yönteminin bir dışsal hiper parametresi var.
# Komşuluk sayısı hiper parametresi. Bu komşuluk sayısı hiper parametresi değiştirilebilirdir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################
# hiperparametre optimizasyonu kapsamında bu ayarlamamız gereken dışsal parametreleri programtik şekilde en doğru
# olarak nasıl ayarlarız?

knn_model = KNeighborsClassifier()
knn_model.get_params() #n_neighbors = 5

# şimdi amacım bu komşuluk sayısını değiştirerek olması gereken en optimum komşuluk sayısının ne olacağını bulmak.
# bunun için parametresi listesi oluşturuyoruz.

knn_params = {"n_neighbors": range(2,50)} # 2 den 50'ye sayı oluşturduk

# bunları aramamız gerekir.

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1 ).fit(X, y) # n_jobs = sonuca hızlı gitmek için kullanılır. # rapor için verbose=1

knn_gs_best.best_params_
# 17 geldi . bu 17 komşuluk sayısıyla gelen bu hiperparametre değeriyle bir final modeli kurmam gerekiyor.


################################################
# 6. Final Model
################################################

#hiperparametreyi 17 olarak tespit ettik o yüzden tekrar model kuruyorum!

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

# modeli kurduk test hatasına bakmamız gerekiyor.

# cv = 5 5 katlı 4 ü ile eğit 1 ile test et. komşuluk sayısının bunla hiç alakası yok

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() #0.73 ' tü => 0.76 oldu
cv_results["test_f1"].mean() # 0.59 'du => 0.61 oldu
cv_results["test_roc_auc"].mean() # 0.78 di => 0.81 oldu

# başarıyı nasıl arttırabilirsin! işte böyle!

random_user = X.sample(1)

knn_final.predict(random_user)