################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Exploratory Data Analysis
################################################

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

################################################
# 3. Modeling using CART
################################################

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
y_pred = cart_model.predict(X) # tüm gözlem değerleri için tahmin edilen değerleri hesaplayalım.

#roc eğrisi için 1.sınıfa ait olma olasılıkları gerekiyor.

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# başarımıza bakalım

print(classification_report(y, y_pred))

# AUC skoru
roc_auc_score(y,y_prob)

# değerlerimiz 1 geldi ve başarımı daha doğru nasıl değerlendirebilirim?

############################
# Holdout Yöntemi ile Başarı Değerlendirme
############################

# bağımsız değişkeni eğit-test et | bağımlı değişkeni eğit-test et
# train_test_split() fonksiyonu, veri setini eğitim ve test setlerine bölmek için kullanılır.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=17)
#train ile model kuruyoruz, test ile test ediyoruz.

#model kuralım
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# 1) train hatasına bakabiliriz 2) test hatasına bakabiliriz


#Train Hatası

y_pred = cart_model.predict(X_train) #Train setinin bağımsız değişkenlerini sorup, tahmin edilen değerleri aldım.
# eğitim setindeki verilere dayanarak, modelin tahmin ettiği hedef değişken değerlerini alırız.

y_prob = cart_model.predict_proba(X_train)[:, 1] #Train setinin bağımsız değişkenlerini sorup, AUC için olasılık değerlerini aldım

print(classification_report(y_train, y_pred)) #Classification_reportu train setinin bağımlı değişkeni ve train setinin bağımlı değişkeni için yaptığım tahminler üzerinden hesapladım
# modelin eğitim seti üzerindeki gerçek hedef değişken değerleri (y_train) ile tahmin ettiği hedef değişken değerleri (y_pred) arasındaki performansı değerlendirir.

roc_auc_score(y_train, y_prob) #AUC hesapladım.
# Bu, modelin ROC eğrisi altındaki alanını (AUC) hesaplar.
# y_train gerçek hedef değişken değerlerini, y_prob ise pozitif sınıf olasılıklarını içerir.

# AUC skorum 1
# traindeki başarım 1


# Test Hatası

y_pred = cart_model.predict(X_test)
# Train setimizi X_train üzerinden kurduk. Şimdi modelin hiç görmediği test setinin bağımsız değişkenlerini gönderdim ve y_ predi aldım.

y_prob = cart_model.predict_proba(X_test)[:,1] # auc için olasılıklar alıyorum

print(classification_report(y_test, y_pred))

roc_auc_score(y_test, y_prob)

## Gözlemliyorum!
# Model eğitildiği veride çok yüksek performans gösterdi ama görmediği bir veriyi sorduğumuzda precision değeri, recall değeri, f1 skoru neredeyse yarı yarıya fark etti.
# Neden? OVERFIT oldu. Veri setini ezberlemiş.
# Çapraz doğrulamaya gitmem lazım!


############################
# CV ile Başarı Değerlendirme
############################

cart_model = DecisionTreeClassifier(random_state=17).fit(X,y) # fit(X,y) yazılsa da olur , yazılmasa da.

cv_results = cross_validate(cart_model,
                            X,y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7058568882098294
cv_results['test_f1'].mean()
# 0.5710621194523633
cv_results['test_roc_auc'].mean()
# 0.6719440950384347

# sonuç : bütün veriyi modellemek için kullanıp bütün veride test ettiğimizde 1 sonucunu aldık. şüphelendik. doğrulamalıyız.
# sonra train ve test olarak ayırıp holdout yöntemine geçtik. burada train seti 1 çıktı.
# bu problem üzerinde gözlem sayısı az olduğu için train test ayrımında da sorunlar ortaya çıkıyor.
# 5 katlı çapraz doğrulama yöntemi ile baktık.
# kıyaslama noktamız başarının yüksekliği değil, daha doğru sonuca gidiyor olmak.


# MODEL BAŞARIMIZI NASIL ARTTIRABİLİRİZ?

# yeni gözlemler ekleyerek, yeni değişkenler ekleyerek, veri ön işleme işlemlerine dokunarak ya da hiperparametre optimizasyonu yaparak gerçekleştirebiliriz.
# bu problem için dengesiz veri yaklaşımları da başarımızı arttırmaya yardımcı olabilir.

#dengesiz veri yaklaşımı => temelinde gözlem birimlerinin dağılımlarının ( iki sınıf ) oranları çok farklı ( bağımlı değişkendeki sınıfların dağılımları )
# bunları birbirine yaklaştırmaya çalıştığımız yöntemlerdir. azaltarak, artırarak, rastgele örneklem yöntemleri gibi yöntemler kullanarak buradaki dengesizliğin giderilmeye
# çalışıldığı yöntemlerdir.

# hiperparametre optimizasyonu senaryosunda ilerleyerek, CART yöntemi için hiperparametre optimizasyonu yapıp, buradaki hatalarımızı/ başarılarımızı artırmaya çalışacağız.


################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

# Mevcut modelin hiperparametreleri nelerdi? sorusunu get_params() ı kullanarak yanıtlayabiliriz.

cart_model.get_params()

# bizim için önemli olan parametreler => min_samples_split , max_depth

cart_params = {"max_depth": range(1,11),
               "min_samples_split": range(2,20)}

# buradaki aralıkları neye göre vericez? ilgili algoritmanın hiperparametresinin default değerlerine gitmek.
# mesela bu örnekte 2 . 2 ile 20 arasında sayılar yazabiliriz.
# default değerin etrafındaki değerleri deneyebiliriz.

# GridSearchCV metodunu getirerek bu parametrelere göre arama yapılmasını istiyoruz.

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1, #işlemcileri tam performans kullanma ayarı
                              verbose=True).fit(X,y)  #verbose raporlama yapmasını istediğimizi True yada 1 ile ifade ederiz.

# Fitting 5 folds for each of 180 candidates, totalling 900 fits
#yani bu iki hiperparametrenin olası 180 kombinasyonu varmış, her birisiyle model kurduğunda(5 katlı ile) bunun hatasını da hesaplayacağından dolayı toplam 900 fit etme işlemi gerçekleşmiş

# en iyi parametreleri getir
cart_best_grid.best_params_

# bu en iyi değerlere karşılık en iyi skor hangisidir?

cart_best_grid.best_score_ #

random = X.sample(1, random_state=45)
# BU VERİ SETİ İÇERİSİNDEN DEĞİŞKENLERİ STANDARTLAŞTIRMADIK AĞAÇ YÖNTEMLERİNDE STANDARTLAŞTIRMAYA İHTİYACIMIZ YOK.
# ÇÜNKÜ BAĞIMSIZ DEĞİŞKENLER KÜÇÜKTEN BÜYÜĞE SIRALANDIKTAN SONRA DEĞERLERİNDEN BÖLÜNÜYOR

cart_best_grid.predict(random)

################################################
# 5. Final Model
################################################

# final model oluşturmanın 1. yolu

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X,y)

cart_final.get_params()

# final model oluşturmanın 2. yolu

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(cart_final,
                            X,y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# 6. Feature Importance
################################################

cart_final.feature_importances_ # değişkenlerin önem düzeyleri


# bu fonksiyon elimizdeki veri setindeki, modeldeki değişkenlerin önem düzeylerini görselliyor
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final, X, num=5)

################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################

# overfiti nasıl anlarız train ve test setinin ayrışmaya başladığı noktadır.
# önüne geçmek için model karmaşıklığının azaltılabileceğini ifade etmiştik.
# model karmaşıklığı arttıkça train setinin hatasının düştüğünü gözlemlemiştik.


train_score, test_score = validation_curve(cart_final, X,y,
                                           param_name="max_depth",
                                           param_range=range(1,11),# 1 ile 11 arasındaki derinliği incelesin ve raporlasın
                                           scoring="roc_auc",
                                           cv=10) # 5 de olur ( 9 ile train et 1 ile test et )

# cart_final model nesnesi kullanılarak ( x ve y için) maksimum derinlik için çeşitli değerlendirmeler yapıldı ve bunların train/test skorları kaydedildi.
# 10 array var(max_depthin parametre sayısı) arrayin içindekiler de cross validation sonucları
# buradaki arrayler bir parametre değerine karşılık elde edilen hataları ifade ediyor.

# bu çıktıların ortalamasını almak istersem;

mean_train_score = np.mean(train_score,axis=1) # 1 array dönecek
mean_test_score = np.mean(test_score, axis=1) # 1 array dönecek

# train ve test hatası birlikte görselleştirilir, ayrım noktaları üzerinden karar verilmeye çalışılır.

plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g') # test ile validasyon aynı şey

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])

################################################
# 8. Visualizing the Decision Tree
################################################

# conda install graphviz
import graphviz

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()

################################################
# 9. Extracting Decision Rules
################################################

# Karar ağaçları: belirli karar, kuralları türetiyor dedik.
# karar kurallarını konsolda gözlemlenebilecek bir görsel tarzda getirme:

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)


################################################
# 10. Extracting Python Codes of Decision Rules
################################################

#skomple yöntemi

# bunları uygulayarak bir ağaç yöntemini canlı sistemlere entegre etme işini yapacağız
# eğer karar kurallarını çıkarıp bir SQL sorgusu haline getirebilirsek aslında database seviyesinde bir model deployment, modeli canlıya alma işlemini tamamlamış oluruz.

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))


################################################
# 11. Prediction using Python Codes
################################################

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)

# python kodunu yazdığımız için dataFrame ' e çevirmedik ama aşağıda df e çevir - predict metodunu kullandığımız için -


################################################
# 12. Saving and Loading Model
################################################

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl") # yüklediğim modeli kullanma

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)