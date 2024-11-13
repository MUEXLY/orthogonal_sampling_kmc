from os.path import exists
from os import walk, mkdir
import lzma


def compress_file(filename: str):
    """
    Compress a file using lzma compression
    """
    with open(f'./dumps/{filename}', 'rb') as f:
        data = f.read()
    compressed_data = lzma.compress(data)
    with open(f'./dumps_compressed/{filename}.xz', 'wb') as f:
        f.write(compressed_data)


def decompress_file(filename: str):
    """
    Decompress a file using lzma compression
    """
    with open(f'./dumps_compressed/{filename}.xz', 'rb') as f:
        compressed_data = f.read()
    data = lzma.decompress(compressed_data)
    with open(f'./dumps/{filename}', 'wb') as f:
        f.write(data)


def main():
    user_input = ''
    while user_input != 'c' and user_input != 'd' and user_input != 'q':
        user_input = input('Compress or decompress? (c/d/q): ')
        if len(user_input) >= 1:
            user_input = user_input[0].lower()
    if user_input == 'q':
        return
    if not exists('dumps'):
        mkdir('dumps')
    if user_input == 'c':
        for (_, _, filename) in walk('dumps'):
            for f in filename:
                if not exists(f'{f}.xz'):
                    print(f'Compressing {f}')
                    compress_file(f)
        return
    for (_, _, filename) in walk('./dumps_compressed'):
        for f in filename:
            if len(f) > 3 and f.endswith('.xz') and not exists(f'./dumps/{f[:-3]}'):
                print(f'Decompressing {f}')
                f = f[:-3]
                decompress_file(f)


if __name__ == '__main__':
    main()
