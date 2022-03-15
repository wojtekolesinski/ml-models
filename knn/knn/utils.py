from typing import List


def read_csv(path: str, delimiter: str, datatypes: list=None, headers: bool=False) -> List[list]:
    with open(path, 'r') as file:
        if not datatypes:
            return [line.strip().split(delimiter) for line in file.readlines()]

        if headers:
            data = [line.strip().split(delimiter) for line in file.readlines()[1:]]
        else:
            data = [line.strip().split(delimiter) for line in file.readlines()]

        for row in data:
            for i in range(len(row)):
                row[i] = row[i].replace('"', '')
                row[i] = datatypes[i](row[i])

        return data







