"""
AÇIKLAMA: 
Bu kod binary classification(kedi,köpek) yapiyor. 20ser adet kedi ve kopek resmi bulup egitimi yaptım egitimi tamamladiktan sonra 
15er adet kedi ve kopek fotografi bulup bu fotograflari siniflandirarak test islemini gerceklestirdim. Bu test sonucunda 15 adet kedi fotografindan 10 adet fotografin
 kedi oldugunu bildi. 15 adet kopek fotografindan da 11 adedinin köpek olduğunu bildi.
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator                                                   # keras kütüphanesini kullandığım için tensorflowu ekledim.

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range =0.2, zoom_range =0.2, horizontal_flip = True)

# resim boyutlarını 64x64 boyutuna getirerek tüm fotoğrafları eşit boyuta getirmeyi ve süreyi kısaltmayı amaçladım.
training_set =train_datagen.flow_from_directory(r'C:\Users\asus\Desktop\CNN\cnn', target_size = (64,64), batch_size = 32, class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set =test_datagen.flow_from_directory(r'C:\Users\asus\Desktop\CNN\cnn', target_size = (64,64), batch_size =32, class_mode = 'binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))          # convolutional işlemi
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                  # max pooling işlemi
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))                                # ikinci convolutional katmanı 
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))                                                  # ikinci max pooling katmanı 
cnn.add(tf.keras.layers.Flatten())                                                                         # flatten işlemi yaparak verileri düzleştirdim.
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))                                               # çıkış katmanı sigmoid 0 ve 1 ler olarak çıkıyor. output 1 ise köpek, 0 ise kedi dir.

cnn.compile(optimizer = 'adam', loss ='binary_crossentropy',metrics = ['accuracy'])
cnn.fit(x =training_set,validation_data = test_set,epochs=10)                                              # compile işlemi yapıp accuracy yani doğruluk değerini öğrenmiş oluruz.


# eğitim tamamlandı şimdi sıra tahmin ederek sınıflandırma işlemi yapmada. 
import numpy as np
import keras.utils as image

test_image = image.load_img(r'C:\Users\asus\Desktop\CNN\test_img\32.jpg',target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image =np.expand_dims(test_image,axis=0)

prediction = cnn.predict(test_image)
training_set.class_indices

if prediction[0][0] == 1:
    print(" Tahmin: Resimdeki hayvan köpek ")
else:
    print(" Tahmin: Resimdeki hayvan kedi ")
    
    
    
    
    
    
    
    
    
    
    
    
    