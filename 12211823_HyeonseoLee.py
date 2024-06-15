import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# 데이터 불러오고 전처리
ratings = pd.read_csv('ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
# 사용자-영화 평점 행렬 생성함
user_mat = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
# 유저 클러스터링 (K-Means 사용함)
kmeans = KMeans(n_clusters=3, random_state=0).fit(user_mat)
user_cluster = kmeans.labels_

# 그룹 추천 알고리즘 선언 구현 부분
# AU 파트 선언
def additive_utilitarian(grouprating):
    return grouprating.sum(axis=0)
# AVG 파트 선언
def average(grouprating):
    return grouprating.mean(axis=0)
# SC 파트 선언
def simple_count(grouprating, threshold=None):
    counts = np.count_nonzero(grouprating, axis=0)
    if threshold is None:
        return pd.Series(counts, index=grouprating.columns)
    else:
        return pd.Series((grouprating > threshold).sum(axis=0), index=grouprating.columns)
# AV 파트 선언
def approval_voting(grouprating, threshold=4):
    return simple_count(grouprating, threshold)
# BC 파트 선언
def borda_count(grouprating):
    ranks = grouprating.rank(axis=1, method='dense', ascending=False)
    ranks = ranks.fillna(0)
    return ranks.sum(axis=0)
# CR 파트 선언
def copeland_rule(grouprating):
    n_users, n_items = grouprating.shape
    copeland = np.zeros(n_items)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            good_i = np.sum(grouprating[:, i] > grouprating[:, j])
            good_j = n_users - good_i
            copeland[i] += (good_i - good_j)
            copeland[j] += (good_j - good_i)
    return copeland

# 5. 각 클러스터 그룹별 추천 결과 출력
for cluster in range(3):
    group_users = user_mat.index[user_cluster == cluster]
    grouprating = user_mat.loc[group_users]

    print(f"cluster {cluster}:")

    # AU 파트
    au = additive_utilitarian(grouprating)
    top10au = au.nlargest(10).index.tolist()
    print("additive utilitarian top 10 Recommend:")
    print(top10au)

    # AVG 파트
    avg = average(grouprating)
    top10avg = avg.nlargest(10).index.tolist()
    print("average top 10 Recommend:")
    print(top10avg)

    # SC 파트
    sc = simple_count(grouprating)
    top10sc = sc.nlargest(10).index.tolist()
    print("simple count top 10 Recommend:")
    print(top10sc)

    # AV 파트
    av = approval_voting(grouprating)
    top10av = av.nlargest(10).index.tolist()
    print("approval voting top 10 Recommend:")
    print(top10av)

    # BC 파트
    bc = borda_count(grouprating)
    top10bc = bc.nlargest(10).index.tolist()
    print("borda count top 10 Recommend:")
    print(top10bc)

    # CR 파트
    cr = copeland_rule(grouprating.values)
    top10cr = pd.Series(cr).nlargest(10).index.tolist()
    print("copeland rule top 10 Recommend:")
    print(top10cr)

    print("\n")