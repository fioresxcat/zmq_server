import numpy as np
import datetime
import time


L2_SIMILARITY_THRESHOLD = 1e6
COSINE_SIMILARITY_THRESHOLD = 1e6

def time2datetime(time):
    return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')


def l2_similarity(f1, f2):
    if isinstance(f1, list):
        f1 = np.array(f1)
    if isinstance(f2, list):
        f2 = np.array(f2)
    return np.sum((f1 - f2) ** 2)


def cosine_similarity(f1, f2):
    if isinstance(f1, list):
        f1 = np.array(f1)
    if isinstance(f2, list):
        f2 = np.array(f2)
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def check_l2_similarity(input_feature: list, table: dict, threshold=L2_SIMILARITY_THRESHOLD) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ở bảng table
    min_distance = 1e9
    min_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        for feature in data['features']:
            distance = l2_similarity(input_feature, feature)
            if distance < min_distance:
                min_distance =  distance
                min_id = id
    
    # nếu khoảng cách gần nhất nhỏ hơn ngưỡng cho phép => 2 đối tượng này là 1
    if min_distance < threshold:
        return min_id
    else:
        return -1


def check_cosine_similarity(input_feature: list, table: dict) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ở bảng table
    max_similarity = -1
    max_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        for feature in data['features']:
            similarity = cosine_similarity(input_feature, feature)
            if similarity > max_similarity:
                max_similarity =  similarity
                max_id = id
    
    # print('max similiarity: ', max_similarity)
    return max_id


def check_cosine_similarity_2(input_feature: list, table: dict) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ở bảng table
    max_similarity = -1
    max_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        sum_similarity = 0
        for feature in data['features']:
            similarity = cosine_similarity(input_feature, feature)
            sum_similarity += similarity
        mean_similarity = sum_similarity / len(data['features'])
        if mean_similarity > max_similarity:
            max_similarity =  mean_similarity
            max_id = id
    
    # print('max similiarity: ', max_similarity)
    return max_id   