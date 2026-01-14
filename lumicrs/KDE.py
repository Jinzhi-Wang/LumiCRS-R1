import json
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


# 读取 JSONL 文件的函数
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# 提取电影名称并统计出现次数
def extract_movie_counts(data):
    movie_counts = Counter()

    for entry in data:
        for message in entry.get("messages", []):
            movies = message.get("movie", [])
            movie_counts.update(movies)  # 统计每部电影的出现次数

    return movie_counts


# 加载训练集和测试集的数据
train_data = load_jsonl('/media/wjz/2TBverydiao/ECR/src_emo/data/redial/train_data_dbpedia_emo.jsonl')
test_data = load_jsonl('/media/wjz/2TBverydiao/ECR/src_emo/data/redial/test_data_dbpedia_emo.jsonl')

# 提取电影出现次数
train_movie_counts = extract_movie_counts(train_data)
test_movie_counts = extract_movie_counts(test_data)

# 获取电影出现次数的列表
train_movie_frequencies = list(train_movie_counts.values())
test_movie_frequencies = list(test_movie_counts.values())

# 绘制KDE
sns.kdeplot(train_movie_frequencies, label='Train', fill=True)
sns.kdeplot(test_movie_frequencies, label='Test', fill=True)
plt.legend(title='Dataset')
plt.title("KDE of Movie Frequencies from Train and Test Datasets")
plt.xlabel("Movie Frequency")
plt.ylabel("Density")
plt.show()
