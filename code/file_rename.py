import os

if __name__ == '__main__':
    path = '../data/RGB/result'
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        folder_name = folder_path.split("\\")[-1]
        #print(folder_path.split("\\")[-1])
        # 獲取資料夾內的所有文件
        files = os.listdir(folder_path)
        
        # 過濾需要重命名的文件
        files_to_rename = [f for f in files if f.startswith(folder_name) and f.endswith('.png')]

        # 開始重命名
        for file in files_to_rename:
            # 提取原文件的序號
            original_number = file.split('_')[-1].split('.')[0]
            
            # 生成新文件名
            new_name = f"{original_number}.png"
            
            # 獲取完整的原文件路徑和新文件路徑
            original_path = os.path.join(folder_path, file)
            new_path = os.path.join(folder_path, new_name)
            
            # 重命名文件
            os.rename(original_path, new_path)
            print(f"Renamed {file} to {new_name}")

        print("重命名完成")