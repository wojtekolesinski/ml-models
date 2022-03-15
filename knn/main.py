from sys import argv

from knn import read_csv, Classifier


print(*read_csv('data/iris.csv', ',', headers=True, datatypes=[float, float, float, float, str]), sep='\n')


# def main():
#     if len(argv) != 2:
#         print("main.py path_to_data category_index")
#
#     data = read_csv(argv[1])
#
#
# if __name__ == '__main__':
#     main()