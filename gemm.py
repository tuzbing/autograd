
import numpy as np
import math

def is_power_of_2(n):
	return ((n & (n - 1)) == 0) and n != 0

def next_power_of_2(n: int) -> int:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return 1 << (n - 1).bit_length()

def get_fp16_gemm_config(M, N, K, with_accum=False, a_transpose=False, b_transpose=False, dtype="fp16"):
	tm, tn, tk = (M+15)//16, (N+15)//16, K//16
	if tm*tn<2048 and tk >= 8 and K > max(M, N):
		tm, tn = (M+31)//32, (N+31)//32
		tk = K/16
		bm = 16 if M < 40 else 32
		bn = 16 if N < 40 else 32
		bk = 16
		wm, wn = 2, 2
		max_k_pow2 = 1 << ((K // bk).bit_length() - 1)
		split_k_partitions = min(next_power_of_2(max(2, int(tk / (tm * tn)))), 32, max_k_pow2)
		split_k_partition_stride = M*N
		gemm_k_iters = (K//bk)//split_k_partitions
		split_k_partition_size = gemm_k_iters * bk
		mn_aligned = (M % bm == 0) and (N % bn == 0)
		k_aligned = K % bk == 0
		lda = M if a_transpose else K
		ldb = K if b_transpose else N
		ldd = N
		tn2, tm2 = (N + bn - 1) // bn, (M + bm - 1) // bm
		grid_x = tn2
		grid_y = tm2
		grid_z = split_k_partitions

		return dict(is_splitk=True, a_transpose=a_transpose, b_transpose=b_transpose, dtype=dtype,
					mn_aligned=mn_aligned, k_aligned=k_aligned,
					bm=bm, bn=bn, bk=bk, wm=wm, wn=2,
					tm=tm2, tn=tn2, lda=lda, ldb=ldb, ldd=ldd,
					split_k_partitions=split_k_partitions,
					split_k_partition_size=split_k_partition_size,
					split_k_partition_stride=split_k_partition_stride,
					k_iter_aligned=gemm_k_iters,
					grid_x=grid_x, grid_y=grid_y, grid_z=grid_z)
	else:
		if dtype == "fp32":
			bm, bn, bk = 32, 64, 16
		else:
			bm, bn, bk = 64, 64, 16
		wm, wn = 2, 2
		tn, tm = (N+bn-1)//bn, (M+bm-1)//bm 
		m_aligned = int(M % bm == 0)
		n_aligned = int(N % bn == 0)
		k_aligned = int(K % bk == 0)
		gemm_k_iters = K // bk
		# swizzle_log = 0
		swizzle_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm>=4 and tn>=4) else (1 if (tm>=2 and tn>=4) else 0))
		sw = 1 << swizzle_log
		lda = M if a_transpose else K
		ldb = K if b_transpose else N
		ldd = N
		grid_x = tn * sw
		grid_y = (tm + sw - 1) // sw
		grid_z = 1

		return dict(is_splitk=False, a_transpose=a_transpose, b_transpose=b_transpose, dtype=dtype,
					mn_aligned=False, m_aligned=m_aligned, n_aligned=n_aligned, k_aligned=k_aligned,
					bm=bm, bn=bn, bk=bk, wm=wm, wn=2,
					tm=tm, tn=tn, swizzle_log=swizzle_log,
					lda=lda, ldb=ldb, ldd=ldd,
					k_iter_aligned=gemm_k_iters,
					grid_x=grid_x, grid_y=grid_y, grid_z=grid_z)

def fp16_gemm_kernel_name(cfg, with_accum, prefix=None):
	prefix = prefix or cfg.get("dtype", "fp16")
	if cfg["is_splitk"]:
		# 仅支持累加, 不支持覆盖
		return (f"{prefix}_gemm_splitk_"
				+ ("t" if cfg['a_transpose'] else "n")
				+ ("t" if cfg['b_transpose'] else "n")
				+ ("t" if cfg['mn_aligned'] else "n" if cfg["is_splitk"] else "")
				+ ("t" if cfg['k_aligned'] else "n" if cfg["is_splitk"] else "")
				+ "_" +
				f"bm{int(cfg['bm'])}_"
				f"bn{int(cfg['bn'])}_"
				f"bk{int(cfg['bk'])}_"
				f"wm{int(cfg['wm'])}_"
				f"wn{int(cfg['wn'])}"
			)
	return (f"{prefix}_gemm_"
			+ ("t" if cfg['a_transpose'] else "n")
			+ ("t" if cfg['b_transpose'] else "n")
			+ "_" +
			f"bm{int(cfg['bm'])}_"
			f"bn{int(cfg['bn'])}_"
			f"bk{int(cfg['bk'])}_"
			f"wm{int(cfg['wm'])}_"
			f"wn{int(cfg['wn'])}_"
			f"a{int(bool(with_accum))}"
		)

