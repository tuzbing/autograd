
import numpy as np
import math

# ---------------- 硬件常量 ----------------
TG_CAPACITY = 32768 # 32KB
SIMD_WIDTH = 32
MAX_SG_PER_TG = 32 # 1024线程/32
NUM_GPU_CORES = 8
GPU_CLOCK = 1.296e9
SCALE_GROUP = 128
BLOCK_K = 128
FRAG = 8
FRAG_BYTES = FRAG * FRAG * 2
TG_ALIGN = 16
ATOMIC_UINT_BYTES = 4

# ---------------- 候选 ----------------
RM_RN_CANDIDATES    = ((2, 4), (4, 4))
PAIR_RATIO_CANDIDATES = (1,)
MAX_WRITE_REG_SLOTS = 16        # pair_ratio * rm * rn 上限
RING_BUF            = 1         # ring 大部分时间为空,rb>1 没有收益
BLOCK_M_CANDIDATES  = (8, 16, 32, 64, 128)
WSG_CANDIDATES = (1,)

# ---------------- 需要用 Xcode 回填的常数 ----------------
R0_WORDS   = 42      # 由 4x4 的 PSO=576 反推 (32, 44]


ETA = {
    1: 1.000,
    2: 1.023,
    3: 1.026,
    4: 1.048,
    5: 1.206,
    6: 1.614,
    7: 1.400,
}
ETA_MAX_R = max(ETA)

def _ceil_div(a, b):
	return (a + b - 1) // b

def _align(n, alignment=TG_ALIGN):
	return _ceil_div(n, alignment) * alignment

# def slot_bytes(rm, rn):
#     return rm * rn * 256 + 16

def smem_bytes(csg, rm, rn):
    return _align(csg * RING_BUF * (rm * rn * 256 + 20))

# ---------------- 寄存器模型(32-bit word / lane) ----------------
def regs_words_compute(rm, rn):
    # 内核里 U 固定为 4，不再用 (rm + rn) <= 6 切换。
    return 2 * (2 * rm * rn + 4 * (rm + rn) + R0_WORDS)

def max_resident_threads_by_regs(regs16):
    table = [(104, 1024), (112, 896), (128, 832), (144, 704),
             (160, 640), (184, 576), (208, 512), (232, 448), (256, 384)]
    for limit, threads in table:
        if regs16 <= limit:
            return threads
    return 0            # 超表:视为不可行,不兜底

# ---------------- 单次tile耗时 ----------------
def t1_loads(bm, bn, rm, rn, K):
    """r=1 时一个满 tile 的 device simdgroup_load 次数。"""
    subs_m = bm // (FRAG * rm)
    subs_n = bn // (FRAG * rn)
    return subs_m * subs_n * _ceil_div(K, FRAG) * (rm + rn)

def wave_r(ntiles, csg):
    """最慢的 compute SG 所在核上，这个 wave 有几个 compute SG。"""
    counts = [0] * NUM_GPU_CORES
    for s in range(ntiles):
        counts[(s // csg) % NUM_GPU_CORES] += 1
    return max(counts)

def score_launch(total_tiles, P, csg, t1):
    """score = T1 * Σ η(r_t)，只加最慢 SG 实际参加的 wave。"""
    owners = P * csg
    crit = _ceil_div(total_tiles, owners)
    acc = 0.0
    for t in range(crit):
        ntiles = min(owners, total_tiles - t * owners)
        r = wave_r(ntiles, csg)
        acc += ETA[r]
    return t1 * acc, crit

# ---------------- 可行形状:线程/寄存器/smem 联合约束 ----------------
def get_pipeline_shapes(rm, rn):
    words = regs_words_compute(rm, rn)
    thr_cap = max_resident_threads_by_regs(words)
    if thr_cap == 0:
        return []
    shapes = []
    for pr in PAIR_RATIO_CANDIDATES:
        if pr * rm * rn > MAX_WRITE_REG_SLOTS:
            continue
        for wsg in range(1, MAX_SG_PER_TG // (pr + 1) + 1):
            csg = pr * wsg
            threads = (csg + wsg) * SIMD_WIDTH
            smem = smem_bytes(csg, rm, rn)
            if smem > TG_CAPACITY:
                continue
            tgs_per_core = min(thr_cap // threads, TG_CAPACITY // smem)
            if tgs_per_core < 1:
                continue
            r_cap = tgs_per_core * csg
            # 满 wave 的 r = k * csg。超过 r_cap 或 η 表的候选不评分。
            k_max = min(r_cap, ETA_MAX_R) // csg
            if k_max < 1:
                continue
            shapes.append(dict(
                pair_ratio=pr, compute_sg=csg, write_sg=wsg,
                threads=threads, smem=smem, r_cap=r_cap, k_max=k_max))
    return shapes

def _search2(M, N, K):
    cands = []
    for rm, rn in RM_RN_CANDIDATES:
        shapes = get_pipeline_shapes(rm, rn)
        for bm in BLOCK_M_CANDIDATES:
            if bm % (rm * FRAG) or M % bm:
                continue
            for bn in range(rn * FRAG, 257, rn * FRAG):
                if N % bn:
                    continue
                num_m_blocks = _ceil_div(M, bm)
                num_n_blocks = _ceil_div(N, bn)
                total_tiles = num_m_blocks * num_n_blocks
                t1 = t1_loads(bm, bn, rm, rn, K)
                for s in shapes:
                    csg = s["compute_sg"]
                    for k in range(1, s["k_max"] + 1):
                        P = NUM_GPU_CORES * k
                        if k > 1 and P * csg > total_tiles:
                            break
                        score, crit = score_launch(total_tiles, P, csg, t1)
                        cands.append(dict(
                            block_m=bm, block_n=bn, block_k=BLOCK_K,
                            reg_tile_m=rm, reg_tile_n=rn,
                            compute_sg=csg, write_sg=s["write_sg"], ring_buf=RING_BUF,
                            group_k=_ceil_div(K, SCALE_GROUP),
                            num_m_blocks=num_m_blocks, num_n_blocks=num_n_blocks,
                            persist_tgs=P, threads_per_tg=s["threads"],
                            r_cap=s["r_cap"], crit=crit, t1=t1, score=score))
    cands.sort(key=lambda c: (
        c["score"], c["block_m"], c["block_n"],
        c["reg_tile_m"], c["reg_tile_n"], c["compute_sg"], c["persist_tgs"]))
    return cands

def get_fp16_gemm_config(M, N, K, cd_dtype=np.float32, with_accum=False, verbose=True):
	# elem_cd = np.dtype(cd_dtype).itemsize
	cands = _search2(M, N, K)
	if not cands:
		raise ValueError(f"GEMM {M}, {N}, {K} 在当前候选下没有可行配置")
	best = cands[0]
	if verbose:
		for c in cands[:8]:
			print("cand", c["block_m"], c["block_n"],
				"rm", c["reg_tile_m"], "rn", c["reg_tile_n"],
				"csg", c["compute_sg"], "P", c["persist_tgs"],
				"crit", c["crit"], "score", round(c["score"], 1))
		print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in best.items()})
	return best

def fp16_gemm_kernel_name(cfg, with_accum):
	return (f"fp16_gemm_"
		f"bm{int(cfg['block_m'])}_"
		f"bn{int(cfg['block_n'])}_"
		f"bk{int(cfg['block_k'])}_"
		f"rm{int(cfg['reg_tile_m'])}_"
		f"rn{int(cfg['reg_tile_n'])}_"
		f"csg{int(cfg['compute_sg'])}_"
		f"wsg{int(cfg['write_sg'])}_"
		f"rb{int(cfg['ring_buf'])}_"
		f"a{int(bool(with_accum))}"
	)
