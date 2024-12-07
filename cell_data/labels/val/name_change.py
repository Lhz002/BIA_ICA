import os

def rename_txt_files(directory, prefix="t", start_num=0, total_files=92):
    """
    将指定目录下的所有 .txt 文件重命名为指定前缀加上递增的三位数后缀，例如 t001.txt, t002.txt, ...

    参数:
    - directory (str): 目标目录路径。
    - prefix (str): 文件名前缀，默认为 "t"。
    - start_num (int): 开始的数字，默认为 1 对应 t001.txt。
    - total_files (int): 最大重命名文件数量，默认为 91（t001.txt 到 t091.txt）。
    """
    # 获取当前目录下所有的 .txt 文件
    txt_files = [f for f in os.listdir(directory) if f.lower().endswith('.txt')]
    txt_files.sort()  # 按字母顺序排序，确保重命名顺序一致

    if not txt_files:
        print("当前目录下没有 .txt 文件需要重命名。")
        return

    if len(txt_files) > total_files:
        print(f"警告：.txt 文件数量 ({len(txt_files)}) 超过指定的最大数量 ({total_files})。")
        print("一些文件将不会被重命名。")

    for idx, filename in enumerate(txt_files):
        if idx >= total_files:
            print(f"跳过文件: {filename}（超过最大重命名数量）")
            continue

        new_num = start_num + idx
        suffix = f"{new_num:03}"  # 生成三位数后缀，如001, 002, ...
        new_filename = f"{prefix}{suffix}.txt"

        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_filename)

        # 检查目标文件名是否已经存在，避免覆盖
        if os.path.exists(dst):
            print(f"跳过文件: {filename}（目标文件名 {new_filename} 已存在）")
            continue

        try:
            os.rename(src, dst)
            print(f"重命名: {filename} -> {new_filename}")
        except Exception as e:
            print(f"无法重命名 {filename} 到 {new_filename}: {e}")

if __name__ == "__main__":
    # 当前脚本所在的目录
    current_directory = os.getcwd()
    rename_txt_files(current_directory, prefix="t", start_num=0, total_files=92)
