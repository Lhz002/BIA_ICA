import os

def modify_labels(directory):
    """
    将指定目录下所有标签文件中的类别索引修改为0。

    参数:
    - directory (str): 标签文件所在的目录路径。
    """
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                modified_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue  # 跳过空行
                    # 将第一个元素（类别索引）改为0
                    parts[0] = '0'
                    modified_line = ' '.join(parts) + '\n'
                    modified_lines.append(modified_line)
                
                # 将修改后的内容写回文件
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(modified_lines)
                
                print(f"已修改文件: {file_path}")
            except Exception as e:
                print(f"无法修改文件 {file_path}: {e}")

if __name__ == "__main__":
    # 设置训练和验证标签目录
    train_labels_dir = os.path.join('D:/desktop/Files/Coding/Python/BIA/Cell_track/cell_data/labels/train')
    val_labels_dir = os.path.join('D:/desktop/Files/Coding/Python/BIA/Cell_track/cell_data/labels/val')
    
    print("正在修改训练标签文件...")
    modify_labels(train_labels_dir)
    
    print("正在修改验证标签文件...")
    modify_labels(val_labels_dir)
    
    print("所有标签文件已成功修改。")
