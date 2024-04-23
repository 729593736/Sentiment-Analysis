import pandas as pd


path = 'ratings/'
movies = pd.read_csv(path + 'movies.csv')
print('电影数目：%d' % movies.shape[0])
ratings = pd.read_csv(path + 'ratings.csv')
my_index = ratings.comment.str.len().sort_values().index
ratings.reindex(my_index)


print('用户数据：%d' % ratings.userId.unique().shape[0])
print('评分数目：%d' % ratings.shape[0])


ratings_5 = ratings.reindex(my_index)[(ratings.rating == 5)].iloc[100000:110000]
ratings_3 = ratings.reindex(my_index)[(ratings.rating == 3)].iloc[100000:110000]
ratings_1 = ratings.reindex(my_index)[(ratings.rating == 1)].iloc[100000:110000]
ratings_2 = ratings.reindex(my_index)[(ratings.rating == 2)].iloc[100000:110000]
ratings_4 = ratings.reindex(my_index)[(ratings.rating == 4)].iloc[100000:110000]

# 这个是来自新的电影影评，是不同电影，我们当作测试集看看效果
# ratings_test2 = ratings.reindex(my_index)[(ratings.rating == 2)].iloc[120000:130000]
# ratings_test = ratings.reindex(my_index).iloc[1000000:1010000]


print('很好（5星）数目：%d' % (ratings_5[ratings_5.rating == 5].shape[0]))
print('一般 (3星) 数目：%d' % (ratings_3[ratings_3.rating == 3].shape[0]))
print('很差（1星）数目：%d' % (ratings_1[ratings_1.rating == 1].shape[0]))
print('较好（4星）数目：%d' % (ratings_4[ratings_4.rating == 4].shape[0]))
print('较差（2星）数目：%d' % (ratings_2[ratings_2.rating == 2].shape[0]))
# print('test:较差 (2星) 数目：%d' % (ratings_2[ratings_2.rating == 2].shape[0]))

# ratings_5.iloc[:10000, [0, 2, 4]].to_csv('5.csv')
ratings_5.iloc[:, [4]].to_csv('5.csv')
ratings_3.iloc[:, [4]].to_csv('3.csv')
ratings_1.iloc[:, [4]].to_csv('1.csv')
ratings_4.iloc[:, [4]].to_csv('4.csv')
ratings_2.iloc[:, [4]].to_csv('2.csv')
# ratings_test2.iloc[:, [4]].to_csv('test2.csv')
# 测试集
# ratings_test.iloc[:, [2, 4]].to_csv('test.csv')
# ratings_with_opinions = ratings_test[(ratings_test.rating == 1) | (ratings_test.rating == 5) | (ratings_test.rating == 3)]
# ratings_with_opinions.iloc[:, [2, 4]].to_csv('test.csv')
