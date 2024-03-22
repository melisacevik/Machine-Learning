                                         2 farklı proje yer almaktadır.

                                         1. Telco Churn Prediction
                                         2. FLO - Gözetimsiz Öğrenme ile Müşteri Segmentasyonu


<img width="1380" alt="Screenshot 2024-03-16 at 09 03 28" src="https://github.com/melisacevik/Machine-Learning/assets/113050206/29060377-4409-46a0-885f-50ced7905489">

# Telco Churn Prediction

Bu proje, bir telekom şirketinin müşteri kaybını tahmin etmek için bir makine öğrenimi modeli geliştirmeyi amaçlamaktadır.

## Proje Linki

Proje GitHub deposuna [buradan](https://github.com/melisacevik/Machine-Learning/blob/master/case-study/case-study3.py) erişebilirsiniz.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn

## Kurulum

1. Projenin klonunu alın: `git clone https://github.com/kullanici/adim/adim.git`
2. Gerekli kütüphaneleri yüklemek için: `pip install -r requirements.txt`
3. Jupyter Notebook veya Python ortamınızda `main.ipynb` dosyasını açın.

## Veri Seti

Proje için kullanılan veri seti, Kaliforniya'daki bir telekom şirketinin 7043 müşterisine ait ev telefonu ve İnternet hizmetleri abonelik bilgilerini içerir.

## Amaç

Bu projenin amacı, veri setindeki müşteri bilgileri ve hizmet kullanımıyla müşteri kaybını tahmin edebilecek bir model geliştirmektir.

## Model Oluşturma

1. Veri seti keşfedilir ve özelliklerin analizi yapılır.
2. Eksik veriler ve aykırı değerler işlenir.
3. Özellik mühendisliği adımlarıyla yeni özellikler türetilir.
4. Kategorik değişkenler sayısal formata dönüştürülür.
5. Model seçimi yapılır ve hiperparametre optimizasyonu gerçekleştirilir.
6. En iyi modelin performansı değerlendirilir.



----------------------------------------------------Case Study 2--------------------------------------------------------------------------
   
<img width="1419" alt="flooo" src="https://github.com/melisacevik/Machine-Learning/assets/113050206/b828cc4f-2374-46b5-8c1f-72a97b4e94c3">


# FLO Müşteri Segmentasyonu Projesi

Bu proje, FLO müşterilerini segmentlere ayırmak ve pazarlama stratejilerini belirlemek amacıyla gerçekleştirilmiştir. Proje, K-Means ve Hiyerarşik Kümeleme algoritmalarını kullanarak müşterileri segmentlere ayırmaktadır.

## Veri Seti

Veri seti, müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgileri içerir. Her bir müşteri için çeşitli özellikler ve davranış verileri bulunmaktadır.

# FLO Müşteri Segmentasyonu Projesi

Bu proje, FLO müşterilerini segmentlere ayırmak ve pazarlama stratejilerini belirlemek amacıyla gerçekleştirilmiştir. Proje, K-Means ve Hiyerarşik Kümeleme algoritmalarını kullanarak müşterileri segmentlere ayırmaktadır.

## Aşamalar

### Aşama 1: Veriyi Hazırlama
### Aşama 2: K-Means ile Müşteri Segmentasyonu

1. Değişkenleri standartlaştırınız.
2. Optimum küme sayısını belirleyiniz.
   - Elbow yöntemi ile k-Elbow grafiği kullanılabilir.
3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
4. Her bir segmenti istatistiksel olarak inceleyiniz.

### Aşama 3: Hierarchical Clustering ile Müşteri Segmentasyonu

1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
3. Her bir segmenti istatistiksel olarak inceleyiniz.

## Gereksinimler

Bu proje çalıştırılmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- numpy
- pandas
- matplotlib
- scikit-learn
- yellowbrick


