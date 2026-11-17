
import numpy as np
import math

FP16_QUANT_BUDGET_BYTES = 32 * 1024 * 1024

def is_power_of_2(n):
	return ((n & (n - 1)) == 0) and n != 0

def next_power_of_2(n: int) -> int:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return 1 << (n - 1).bit_length()

def _align_up(n, align):
	return (n + align - 1) // align * align

def fp16_gemm_k_tile(M, N, K, budget_bytes=FP16_QUANT_BUDGET_BYTES):
	meta = _align_up(M*4, 16) + _align_up(N*4, 16)
	avail = max(0, budget_bytes - meta)
	max_k = max(128, (avail // (2*(M+N)*128))*128)
	if K <= max_k:
		return K
	n_chunks = (K + max_k - 1) // max_k
	return _align_up((K + n_chunks - 1) // n_chunks, 128)

def build_fp16_k_tile_configs(M, N, K, a_transpose=False, b_transpose=False, budget_bytes=FP16_QUANT_BUDGET_BYTES):
	kt = fp16_gemm_k_tile(M, N, K, budget_bytes)
	kt_sizes = [kt] + ([K % kt] if K % kt else [])
	return {
		"dtype": "fp16",
		"k_tile": kt,
		"a_transpose": a_transpose,
		"b_transpose": b_transpose,
		"sub_configs": {
			k_sub: get_fp16_gemm_config(M=M, N=N, K=k_sub, a_transpose=a_transpose, b_transpose=b_transpose, dtype="fp16")
			for k_sub in kt_sizes
		},
	}

def get_fp16_gemm_config(M, N, K, with_accum=False, a_transpose=False, b_transpose=False, dtype="fp16", lda=None):
	tm, tn, tk = (M+15)//16, (N+15)//16, K//16
	splitk_limit = 256 # M1 8核满载仅需64*64线程组(tm*tn=768), 避免dx误入低效splitk
	can_splitk = (dtype != "fp16") or (not a_transpose and not b_transpose)
	if can_splitk and tm*tn<splitk_limit and tk >= 8 and K > max(M, N):
		tm, tn = (M+31)//32, (N+31)//32
		tk = K/16
		bm = 32 if dtype == "fp16" else (16 if M < 40 else 32)
		bn = 32 if dtype == "fp16" else (16 if N < 40 else 32)
		bk = 16
		wm, wn = 2, 2
		max_k_pow2 = 1 << ((K // bk).bit_length() - 1)
		split_k_partitions = min(next_power_of_2(max(2, int(tk / (tm * tn)))), 32, max_k_pow2)
		split_k_partition_stride = M*N
		gemm_k_iters = (K//bk)//split_k_partitions
		split_k_partition_size = gemm_k_iters * bk
		mn_aligned = (M % bm == 0) and (N % bn == 0)
		k_aligned = K % bk == 0
		lda = (M if a_transpose else K) if lda is None else lda
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
		if dtype == "fp16":
			bm, bn, bk = 64, 64, 32
		elif M >= 64:
			bm, bn, bk = 64, 64, 16
		else:
			bm, bn, bk = 32, 64, 16

		# if dtype != "fp32" or M >= 64:
		# 	bm, bn, bk = 64, 64, 16
		# else:
		# 	bm, bn, bk = 32, 64, 16
		wm, wn = 2, 2
		tn, tm = (N+bn-1)//bn, (M+bm-1)//bm 
		m_aligned = int(M % bm == 0)
		n_aligned = int(N % bn == 0)
		k_aligned = int(K % bk == 0)
		gemm_k_iters = K // bk
		# swizzle_log = 0
		swizzle_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm>=4 and tn>=4) else (1 if (tm>=2 and tn>=4) else 0))
		sw = 1 << swizzle_log
		lda = (M if a_transpose else K) if lda is None else lda
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

