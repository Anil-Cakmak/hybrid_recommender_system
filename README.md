# hybrid_recommender_system
MIUUL Data Science and Machine Learning Bootcamp Projesi

## İş Problemi
ID'si verilen kullanıcı için item-based ve user-based recommender
yöntemlerini kullanarak 10 film önerisi yapınız.

## Veri Seti Hikâyesi
Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan
derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016
tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar
rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

### movie.csv

**3 Değişken,  27278 Gözlem,  1.5 MB**

**movieId**: Eşsiz film numarası. \
**title**: Film adı. \
**genres**: Filmin türü.

### rating.csv

**4 Değişken,  20000263 Gözlem,  690.4 MB**

**userid**: Eşsiz kullanıcı numarası (UniqueID). \
**movieId**: Eşsiz film numarası (UniqueID). \
**rating**: Kullanıcı tarafından filme verilen puan. \
**timestamp**: Değerlendirme tarihi 

