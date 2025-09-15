import pandas as pd

# 示例数据
user_list = [1, 2, 3]
item_list = [[10, 23, 45, 67], [12, 34], [45, 67, 89]]

# 创建用户-物品对
data = {'user_id': user_list, 'item_list': item_list}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame写入train.txt
with open('train.txt', 'w') as f:
    for _, row in df.iterrows():# iterrows()返回索引和行数据，使用下划线(_)表示不关心索引值
        line = f"{row['user_id']} " + " ".join(map(str, row['item_list'])) + "\n"
        f.write(line)
