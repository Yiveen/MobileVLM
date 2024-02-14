import os
'''
主要是为了解决在目前数据集生成中，有一些rgb图片和seg图片的名字是Image0002.png，segmentation0002.png 的问题
'''
# Function to rename files in each subdirectory
def rename_files(directory):
    for file in os.listdir(directory):
        if file.startswith('Image'):
            os.rename(os.path.join(directory, file), os.path.join(directory, 'Image0001.png'))
        else:
            os.rename(os.path.join(directory, file), os.path.join(directory, 'segmentation0001.png'))

dataset_folder = '/home/yiwen/Projects/MobileVLM/data/vima_test'
for folder in os.listdir(dataset_folder):
    rename_files(os.path.join(dataset_folder, folder))
    print('folder', folder)




