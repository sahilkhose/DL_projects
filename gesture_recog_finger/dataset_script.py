import os
import shutil
from shutil import move

print("--"*40)
print("\nImports Done!!\n")
print("--"*40)


train_dir = './fingers/train'
test_dir = './fingers/test'



def create_folders():
	for i in range(6):
		os.mkdir(os.path.join(train_dir, f"{i}"))

	for i in range(6):
		os.mkdir(os.path.join(test_dir, f"{i}"))


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def create_dataset():
	for file in files(train_dir):
		label = file.replace('.png', '')[-2]
		# hand = file.replace('.png', '')[-1]
		move(os.path.join(train_dir, file), os.path.join(train_dir, f"{label}"))
		# print(label)

	for file in files(test_dir):
		label = file.replace('.png', '')[-2]
		# hand = file.replace('.png', '')[-1]
		move(os.path.join(test_dir, file), os.path.join(test_dir, f"{label}"))
		# print(label)


def main():
    create_folders()
    create_dataset()

if __name__ == "__main__":
    main()