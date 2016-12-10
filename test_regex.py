import re


def main():
    with open('data/test.txt') as flo:
        read_data = flo.read()

    pattern = r'\n(?:(?!(\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2}:\d{2})))'
    # pattern = r''

    re.search(pattern, read_data)
    print(read_data)
    result = re.search(pattern, read_data)
    print(result)

    print(re.sub(pattern, 'INSERT', read_data))


if __name__ == '__main__':
    main()
