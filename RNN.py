"""
Using the my artificial datas I solved the sentimental analysis problem
"""

import numpy as numpy
import pandas as pd 

from gensim.utils import Word2Vec

from tensorflow.keras.models import Sequentiai
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

data = {
    "text": [
        # positive comments
        "Yemekler çok lezzetliydi, özellikle kebap harikaydı.",
        "Garsonlar çok nazik ve güler yüzlüydü.",
        "Ortam çok şık ve huzurluydu, keyif aldım.",
        "Fiyatlar çok uygundu, porsiyonlar oldukça büyüktü.",
        "Sunum harikaydı, tabaklar özenle hazırlanmıştı.",
        "Restoran çok temizdi, hijyen konusunda başarılıydı.",
        "Mekan çok ferah ve rahatlatıcıydı.",
        "Tatlılar mükemmeldi, özellikle cheesecake harikaydı.",
        "Siparişler çok hızlı geldi, beklemeye gerek kalmadı.",
        "İçecekler çok taze ve doğal malzemelerle hazırlanmıştı.",
        "Porsiyonlar çok doyurucuydu, tam fiyatını hak ediyordu.",
        "Müzik çok hoştu, ortamın ambiyansını güzelleştiriyordu.",
        "Lezzet mükemmeldi, yemekler tam kıvamındaydı.",
        "Servis çok hızlı ve profesyoneldi.",
        "Tatlı menüsü oldukça geniş ve çeşitliydi.",
        "Yemekler sıcak ve tam zamanında servis edildi.",
        "Garsonlar çok ilgililerdi, her ihtiyacımızı karşıladılar.",
        "Mekan çok modern ve şık bir tasarıma sahipti.",
        "Etler tam istediğim kıvamda pişirilmişti.",
        "Menüde çok fazla seçenek vardı, herkes kendine göre bir şey bulabiliyordu.",
        "Ortam çok sıcak ve samimiydi, tekrar gelmek isterim.",
        "Çalışanlar çok güler yüzlüydü, hizmet kalitesi harikaydı.",
        "Tatlılar çok tazeydi, özellikle sufle enfesti.",
        "Yemekler doyurucu ve çok lezzetliydi.",
        "Fiyat-performans açısından çok başarılı bir restorandı.",

        # negative comments
        "Yemekler çok soğuk geldi, hiç beğenmedim.",
        "Garsonlar ilgisizdi, siparişimizi bile zor verdik.",
        "Ortam çok gürültülüydü, sohbet etmek imkansızdı.",
        "Fiyatlar çok yüksekti, bu kaliteye kesinlikle değmezdi.",
        "Masalar çok kirliydi, hijyen konusunda başarısızdı.",
        "Servis çok yavaştı, sipariş için uzun süre bekledik.",
        "Tatlılar bayattı, özellikle pasta hiç taze değildi.",
        "Menü çok sıradandı, farklı lezzetler beklerken hayal kırıklığı yaşadım.",
        "Yemekler çok yağlı ve ağırdı, midemi rahatsız etti.",
        "Sipariş eksik geldi, beklemek zorunda kaldık.",
        "Garson siparişi yanlış getirdi, değişim için uğraştık.",
        "Yemekler aşırı tuzluydu, yenilebilecek gibi değildi.",
        "Fiyatlar gerçekten uçuktu, porsiyonlar ise çok küçüktü.",
        "Tatlılar çok şekerliydi, yemesi zor oldu.",
        "Yemeklerin sunumu çok basitti, özensizdi.",
        "Mekan çok havasız ve sıkıcıydı.",
        "Müşteri hizmetleri çok kötüydü, ilgilenmediler.",
        "Müzik çok yüksek sesliydi, rahatsız ediciydi.",
        "Lezzet konusunda çok başarısızdı, yemekleri yiyemedim.",
        "Porsiyonlar inanılmaz küçüktü, kesinlikle doyurucu değildi.",
        "Hijyen kurallarına uyulmamıştı, masalar çok pisti.",
        "Etler tam pişmemişti, çok rahatsız ediciydi.",
        "Servis aşırı yavaştı, yemeğimizin gelmesi çok uzun sürdü.",
        "Yemekler çok kötüydü, bir daha asla gelmem."
    ],
    "label": [
        "positive", "positive", "positive", "positive", "positive", 
        "positive", "positive", "positive", "positive", "positive", 
        "positive", "positive", "positive", "positive", "positive", 
        "positive", "positive", "positive", "positive", "positive", 
        "positive", "positive", "positive", "positive", "positive", 
        
        "negative", "negative", "negative", "negative", "negative", 
        "negative", "negative", "negative", "negative", "negative", 
        "negative", "negative", "negative", "negative", "negative", 
        "negative", "negative", "negative", "negative", "negative", 
        "negative", "negative", "negative", "negative"
    ]
}
data["text"],data["label"]=shuffle(data["text"],data["label"],random_state=49)


df=pd.DataFrame(data)


tokenizer=Tokenizer()
tokenizer.fit_on_text(df["text"])
sequences=tokenizer.texts_to_sequence(df["text"])
word_index= tokenizer.word_index

maxlen=max(len(seq) for seq in sequences)
X=pad_sequence(sequences,maxlen)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(df["label"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sentences= [text.split() for text in df["text"]]
word2_model=Word2Vec(sentences,vector_size=50,window=5,min_count=1,sg=0)

embedding_dim=50
embedding_matrix= np.zeros((len(word_index)+1,embedding_dim))

for word,i in word_index.items():
    if word in word2_model.wv[i]:
        embedding_matrix[i]=word2_model.wv[word]

model= Sequentiai()

model.add(Embedding(input_din=len(word_index)+1,output_dim=embedding_dim,weights=[embedding_matrix],input_length=maxlen))

model.add(SimpleRNN(units=1,return_sequences=False))

model.add(Dense(1,activation="Sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=20,batch_size=2,validation_data=(X_test,y_test))


def classify_sentence(sentence):
    seq=tokenizer.texts_to_sequence(sentence)
    pad_seq=pad_sequence(seq,maxlen)
    prediction=model.predict(pad_seq)
    predicted_class=(prediction>0.5).astype(int)
    label="positive" if predicted_class[0][0] else "negative"

    return label

