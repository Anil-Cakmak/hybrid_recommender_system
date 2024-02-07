import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

########################################
# User-Movie dataframe'inin oluşturulması
#########################################

movie = pd.read_csv("recommendation_systems/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("recommendation_systems/datasets/movie_lens_dataset/rating.csv")
df = rating.merge(movie, how="inner", on="movieId")

# Az yorum alan filmler üzerinden anlamlı bir ilişki kurulamayacağından ve hesaplama maliyetlerini azaltmak adına
# bu filmlerin veri setinden çıkarılması.
comment_counts = pd.DataFrame(df.title.value_counts())
rare_movies = comment_counts[comment_counts["count"] < 1000].index
common_movies = df[~df.title.isin(rare_movies)]

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


# Veriye hazırlık işlemlerinin fonksiyonlaştırılması.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("recommendation_systems/datasets/movie_lens_dataset/movie.csv")
    rating = pd.read_csv("recommendation_systems/datasets/movie_lens_dataset/rating.csv")
    df = rating.merge(movie, how="inner", on="movieId")
    comment_counts = pd.DataFrame(df.title.value_counts())
    rare_movies = comment_counts[comment_counts["count"] < 1000].index
    common_movies = df[~df.title.isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

# Tavsiye yapılacak kullanıcı.
random_user = pd.Series(user_movie_df.index).sample(1, random_state=10).values[0]

##########################################
# USER BASED RECOMMENDATION
##########################################

#############################################
# Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

random_user_comments = user_movie_df.loc[random_user]
movies_watched = random_user_comments[random_user_comments.notnull()].index.values.tolist()

#############################################
# Benzer kullanıcıların belirlenmesi.
#############################################

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()

# Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların belirlenmesi.
threshold = len(movies_watched)*0.6
users_same_movies = user_movie_count[lambda x: x >= threshold].index.values.tolist()

#############################################
# Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

final_df = movies_watched_df.loc[users_same_movies]

corr_df = final_df.T.corr().unstack().drop_duplicates().sort_values(ascending=False)
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId_1", "userId"]
corr_df.reset_index(inplace=True)

# Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıların filtrelenmesi.
top_users = corr_df[(corr_df["userId_1"] == random_user) & (corr_df["corr"] > 0.65)][["userId", "corr"]].\
    reset_index(drop=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')


#############################################
# Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Her bir film kırılımında weighted_rating ortalamasının alınması.
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})

# recommendation_df içerisinden weighted rating'i en yüksek olan 5 filmin seçilmesi.
movies_to_be_recommend = recommendation_df.sort_values(by="weighted_rating", ascending=False).\
    head(5).index.values.tolist()

# Tavsiye edilen 5 film.
recommended_movies = movie[movie.movieId.isin(movies_to_be_recommend)]["title"].values.tolist()

# Burden of Dreams (1982)
# Inn of the Sixth Happiness, The (1958)
# King of Hearts (1966)
# Early Summer (Bakushû) (1951)
# Closely Watched Trains (Ostre sledované vlaky) (1966)

#############################################
# ITEM-BASED RECOMMENDATION
#############################################

# Öneri yapılacak kullanıcının en yüksek puan verdiği filmlerden değerlendime tarihi en güncel olan filmin seçilmesi.
df["timestamp"] = pd.to_datetime(df["timestamp"])
user_df = df[df.userId == random_user]
movie_name = user_df.sort_values(by=["rating", "timestamp"], ascending=False)["title"].iloc[0]
# American Beauty (1999)

# Seçili filmle diğer filmlerin korelasyonunun bulunması.
movie_df = user_movie_df[movie_name]
corr = user_movie_df.corrwith(movie_df).sort_values(ascending=False)

# Tavsiye edilen 5 film.
recommendations = corr[corr.index != movie_name].iloc[0:5].index.tolist()

# Traffic (2000)
# Monster's Ball (2001)
# Woodstock (1970)
# Mystic River (2003)
# Barbarian Invasions, The (Les invasions barbares) (2003)
