import json

# 输入文件路径
file1_path = '/media/wjz/2TBverydiao/ECR/src_emo/data/redial_gen/train_data_processed.jsonl'
file2_path = '/media/wjz/2TBverydiao/ECR/src_emo/test_data_processed_0_9.jsonl'

# 输出文件路径
output_file_path = '/media/wjz/2TBverydiao/ECR/src_emo/data/redial_gen/train_data_processed_0_9.jsonl'

# 阅读第一个JSONL文件
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 阅读文件内容
data1 = read_jsonl(file1_path)
data2 = read_jsonl(file2_path)

# 合并数据
merged_data = data1 + data2

# 写入合并后的JSONL文件
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for entry in merged_data:
        json_line = json.dumps(entry, ensure_ascii=False)
        output_file.write(json_line + '\n')

print(f'合并完成，结果已保存到 {output_file_path}')
