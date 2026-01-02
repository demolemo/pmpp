import numpy as np


def naive_tiled_matmul(a, b, c, N, TILE_SIZE):
    for row_tile_index in range(N // TILE_SIZE):
        for col_tile_index in range(N // TILE_SIZE):

            for tile_index in range(N // TILE_SIZE):
                a_tile = a[row_tile_index * TILE_SIZE:(row_tile_index + 1) * TILE_SIZE, tile_index * TILE_SIZE:(tile_index + 1) * TILE_SIZE]
                b_tile = b[tile_index * TILE_SIZE:(tile_index + 1) * TILE_SIZE, col_tile_index * TILE_SIZE:(col_tile_index + 1) * TILE_SIZE]
                c[row_tile_index * TILE_SIZE:(row_tile_index + 1) * TILE_SIZE, col_tile_index * TILE_SIZE:(col_tile_index + 1) * TILE_SIZE] += a_tile @ b_tile

    assert np.allclose(a @ b, c), 'Tiled matmul works incorrectly'
    print('Tiled matmul works correctly')


def basic_example():
    # basic example to see the code with my own eyes
    a = np.random.rand(4, 4)
    b = np.random.rand(4, 4)
    c = np.zeros_like(a)

    tile_size = 2
    c_canonical = a @ b
    c[0:tile_size, 0:tile_size] += a[0:tile_size, 0:tile_size] @ b[0:tile_size, 0:tile_size]
    c[0:tile_size, 0:tile_size] += a[0:tile_size, tile_size:] @ b[tile_size:, 0:tile_size]
    assert np.allclose(c_canonical[:2, :2], c[:2, :2]), 'Elements differ'


def main():
    # N = int(input())
    # while N not in (8, 16, 32, 64, 128):
    #     print('Please enter N that lies in: [8, 16, 32, 64, 128]: ')
    #     N = int(input())
    
    N = 128
    TILE_SIZE = N // 4
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    c = np.zeros_like(a)
    # tiled_matmul(a, b, c, N, TILE_SIZE)
    # 
    # print('result: ', c)

    # print('canonical result', a @ b)
    basic_example()
    naive_tiled_matmul(a, b, c, N, TILE_SIZE)


if __name__ == "__main__":
    main()


