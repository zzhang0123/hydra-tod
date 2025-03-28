import numpy as np
from scipy.linalg import scalapack
from mpi4py import MPI
import time

# 初始化MPI环境
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 定义矩阵维度 (可调整)
N = 2000  # 矩阵大小 (N x N)
nprow = 2  # 进程网格行数 (需满足 nprow * npcol = size)
npcol = 2  # 进程网格列数

def main():
    if rank == 0:
        print(f"Running ScaLAPACK demo with {size} MPI processes.")

    # 仅在根进程生成全局矩阵A和向量b
    if rank == 0:
        A_global = np.random.rand(N, N).astype(np.float64)
        A_global = A_global + A_global.T  # 确保对称正定
        b_global = np.random.rand(N).astype(np.float64)
    else:
        A_global = np.empty((N, N), dtype=np.float64)
        b_global = np.empty(N, dtype=np.float64)

    # 广播全局矩阵和向量到所有进程
    comm.Bcast(A_global, root=0)
    comm.Bcast(b_global, root=0)

    # 创建ScaLAPACK进程网格
    context = scalapack.blacs_grid((nprow, npcol), order='Row')

    # 将矩阵A和向量b按块分布到进程网格
    A_dist = scalapack.distribute_matrix(A_global, context)  # 分布式存储的A
    b_dist = scalapack.distribute_matrix(b_global.reshape(-1, 1), context)  # 分布式存储的b

    # 记录ScaLAPACK求解时间
    start_time = time.time()

    # 调用ScaLAPACK的求解器 (PDPOSV: 对称正定矩阵的Cholesky分解)
    x_dist, _ = scalapack.pdpotrs(A_dist, b_dist, lower=False)

    # 收集分布式解到根进程
    x_scalapack = scalapack.collect_matrix(x_dist, context)

    scalapack_time = time.time() - start_time

    # 根进程验证结果
    if rank == 0:
        # 串行求解对比
        start_serial = time.time()
        x_serial = np.linalg.solve(A_global, b_global)
        serial_time = time.time() - start_serial

        # 计算误差
        error = np.linalg.norm(x_scalapack.flatten() - x_serial)
        print(f"ScaLAPACK time: {scalapack_time:.3f} s")
        print(f"Serial time:    {serial_time:.3f} s")
        print(f"Error norm:     {error:.3e}")

if __name__ == "__main__":
    main()