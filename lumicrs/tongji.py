import json
import csv
from collections import Counter

# 读取 JSON 文件
with open('/media/wjz/2TBverydiao/ECR/src_emo/save/redial_rec/rec.json', 'r') as file:
    data = [json.loads(line) for line in file.readlines()]

# 提取所有 label_position 的值
label_positions = [entry['label_position'] for entry in data]

# 统计每个 label_position 出现的次数
position_counts = Counter(label_positions)

# 打印出统计结果
print(position_counts)

# 保存到 CSV 文件
with open('label_position_counts.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['label_position', 'count'])  # 写入标题
    for position, count in position_counts.items():
        writer.writerow([position, count])

print("CSV 文件已保存: label_position_counts.csv")
