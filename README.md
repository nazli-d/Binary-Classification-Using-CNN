# Binary Classification (Cat and Dog) Using Convolution Neural Network

Bu proje, kedi ve köpek resimlerini sınıflandırmak için kullanılan bir CNN (Convolutional Neural Network) modeli oluşturur. Eğitim veri seti olarak 20'şer adet kedi ve köpek resmi kullanılmıştır. Eğitim tamamlandıktan sonra, 15'er adet kedi ve köpek fotoğrafı kullanılarak test işlemi gerçekleştirilmiştir. Projede kullanılan model, keras kütüphanesi üzerinde TensorFlow backend'i kullanılarak oluşturulmuştur.

## Gereksinimler

Proje için aşağıdaki kütüphanelere ihtiyaç vardır:
- tensorflow
- keras
- numpy

Kütüphaneleri yüklemek için aşağıdaki komutları kullanabilirsiniz:

``` bash 
pip install tensorflow
pip install keras 
pip install numpy
```

## Veri Seti

Proje için kullanılan veri seti, kedi ve köpek resimlerinin bulunduğu bir dizinden yüklenmektedir. Eğitim ve test veri setleri ayrı ayrı belirlenmiştir. Eğitim veri seti için kullanılan resimler, `cnn` dizini altında bulunmalıdır. Test veri seti için kullanılan resimler ise `test_img` dizini altında bulunmalıdır.

## Modelin Oluşturulması

Proje, `tf.keras.models.Sequential` sınıfı kullanılarak bir CNN modeli oluşturur. Model, aşağıdaki katmanları içermektedir:

- Convolutional katmanı: 32 adet 3x3 filtre ve ReLU aktivasyon fonksiyonu
- Max pooling katmanı: 2x2 boyutunda bir pooling işlemi
- İkinci convolutional katmanı: 32 adet 3x3 filtre ve ReLU aktivasyon fonksiyonu
- İkinci max pooling katmanı: 2x2 boyutunda bir pooling işlemi
- Flatten katmanı: Verileri düzleştiren bir işlem
- Tam bağlantılı (dense) katman: 128 nöron ve ReLU aktivasyon fonksiyonu
- Çıkış katmanı: Sigmoid aktivasyon fonksiyonu ile bir değer (0 veya 1 değeri) döndürür

## Test İşlemi

Test işlemi için `test_img` dizini altında bulunan resimler kullanılır. Bu resimler, `ImageDataGenerator` sınıfı kullanılarak önceden işlenir. Ardından, model üzerinde `predict` işlemi gerçekleştirilir ve sonuç tahmin edilir. Tahmin sonucuna göre, resimdeki hayvanın kedi mi yoksa köpek mi olduğu belirlenir.

## Kullanım

Proje dosyalarını indirdikten sonra, aşağıdaki adımları takip ederek projeyi kullanabilirsiniz:

1. Eğitim ve test veri setlerini uygun dizinlere yerleştirin.
2. Gerekli kütüphaneleri yükleyin.
3. Modelin oluşturulması ve eğitimi için `binary-classification.py` dosyasını çalıştırın.
4. Test işlemi için `CNN` dosyasında bulunan `test_image` dosyasına test etmek istediğiniz resimleri yükleyin.

## Test Görüntüleri

### Köpek fotoğraflarının testi
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-1.jpg)
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-3.jpg)
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-4.jpg)

### Kedi fotoğraflarının testi
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-2.jpg)
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-5.jpg)
![](https://github.com/nazli-d/Binary-Classification-Using-CNN/blob/main/outputs/test-6.jpg)

## Katkıda Bulunma

Bu proje her türlü katkıya açıktır. Katkıda bulunmak için şu adımları takip edebilirsiniz:
1. Bu depoyu (`repository`) çatalayın (fork).
2. Yaptığınız değişiklikleri içeren yeni bir dal (branch) oluşturun.
3. Değişikliklerinizi bu yeni dalda yapın ve düzenleyin.
4. Değişikliklerinizi başka bir dalda test edin.
5. Değişikliklerinizi orijinal depoya (upstream repository) geri göndermek için bir birleştirme isteği (pull request) açın.
