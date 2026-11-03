
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
MAX_TGS_PER_CORE = 7
C_SG_CAL = {(4, 4): 99.0, (2, 4): 176.0}
C_SG       = 98.0
C_ALU      = 6.5
R0_WORDS   = 42      # 由 4x4 的 PSO=576 反推 (32, 44]
T_W_CYC    = 2000.0
WRITER_UTIL_MAX = 0.6

MMA_CYC    = 4.0     # 每个 8x8x8 MMA 占 core 4 cycle(2.65 TFLOPS 峰值推出)
L_CYC      = 387.0   # 一次 device->simdgroup_load 往返延迟,沿用旧标定;K1 后用 2x4 实测重标
BW_EFF     = 33e9    # 读+写合计的有效带宽(Xcode 实测钉在 32~36 GB/s)
DEDUPE     = 2.4     # L1/L2 命中使 DRAM 读量缩小的倍数(dX 实测 2.4,dW 约 3.1)
T_W_BASE   = 600.0   # writer 处理一条消息的 cycle:未实测
T_W_FRAG   = 150.0   # 每个 fragment 的增量:未实测

PREFERRED = {(512, 384): (16, 32), (32768, 384): (128, 128)}


def _ceil_div(a, b):
	return (a + b - 1) // b

def _align(n, alignment=TG_ALIGN):
	return _ceil_div(n, alignment) * alignment

def slot_bytes(rm, rn):
    return rm * rn * 256 + 16

def smem_bytes(compute_sg, ring_buf, rm, rn):
    return _align(compute_sg * ring_buf * (slot_bytes(rm, rn) + ATOMIC_UINT_BYTES))

# ---------------- 寄存器模型(32-bit word / lane) ----------------
def unroll_depth(rm, rn):
    """必须和 kernel 的 U 公式一致"""
    return 4 if (rm + rn) <= 6 else 2

def regs16_compute(rm, rn):
    u = unroll_depth(rm, rn)
    # float8x8 acc:2 word/frag;half8x8 frag:1 word/frag,U 步同时活跃
    return 2 * (2 * rm * rn + u * (rm + rn) + R0_WORDS)

def regs16_write(rm, rn, pr):
    # acc[PR][SUB] 是 float2;另有 sa_v[rm]、sb0[rn]、sb1[rn]
    return 2 * (2 * pr * rm * rn + rm + 2 * rn + R0_WORDS)

def max_resident_threads_by_regs(regs16):
    table = [(104, 1024), (112, 896), (128, 832), (144, 704),
             (160, 640), (184, 576), (208, 512), (232, 448), (256, 384)]
    for limit, threads in table:
        if regs16 <= limit:
            return threads
    return 0            # 超表:视为不可行,不兜底

# ---------------- 消息数(镜像 messages_for_tile) ----------------
def _full_subs(extent, block, rfrags_elems, subs):
    return (extent // block) * subs + min((extent % block) // rfrags_elems, subs)

def count_messages(M, N, block_m, block_n, rm, rn, group_k):
    subs_m = block_m // FRAG // rm
    subs_n = block_n // FRAG // rn
    nmb, nnb = _ceil_div(M, block_m), _ceil_div(N, block_n)
    interior = _full_subs(M, block_m, rm * FRAG, subs_m) * _full_subs(N, block_n, rn * FRAG, subs_n)
    total_subs = nmb * nnb * subs_m * subs_n
    return group_k * (interior + (total_subs - interior) * rm * rn)

# ---------------- 可行形状:线程/寄存器/smem 联合约束 ----------------
def get_pipeline_shapes(rm, rn):
    sub = rm * rn
    shapes = []
    for pr in PAIR_RATIO_CANDIDATES:
        if pr * sub > MAX_WRITE_REG_SLOTS:
            continue
        thr_cap = max_resident_threads_by_regs(max(regs16_compute(rm, rn), regs16_write(rm, rn, pr)))
        if thr_cap == 0:
            continue
        for wsg in range(1, MAX_SG_PER_TG // (pr + 1) + 1):
            csg = pr * wsg
            threads = (csg + wsg) * SIMD_WIDTH
            smem = smem_bytes(csg, RING_BUF, rm, rn)
            if smem > TG_CAPACITY:
                continue
            tgs_per_core = min(thr_cap // threads, TG_CAPACITY // smem)
            if tgs_per_core < 1:                       # 不再 max(1, ...)
                continue
            shapes.append(dict(pair_ratio=pr, compute_sg=csg, write_sg=wsg,
                               threads=threads, smem=smem,
                               tgs_per_core=tgs_per_core, thr_cap=thr_cap))
    return shapes

def _search(M, N, K, elem_cd, with_accum, allow_ragged, verbose):
    group_k = _ceil_div(K, SCALE_GROUP)
    cands = []
    for rm, rn in RM_RN_CANDIDATES:
        shapes = get_pipeline_shapes(rm, rn)
        for bm in BLOCK_M_CANDIDATES:
            if bm % (rm*FRAG) or M % bm:
                continue
            for bn in range(rn*FRAG, 257, rn*FRAG):
                if N % bn:
                    continue
                num_m_blocks = _ceil_div(M, bm)
                num_n_blocks = _ceil_div(N, bn)
                total_tiles = num_m_blocks * num_n_blocks
                mma_tiles = (bm // FRAG) * (bn // FRAG) * _ceil_div(K, FRAG)

                for s in shapes:
                    csg, wsg = s["compute_sg"], s["write_sg"]
                    pair_ratio = s["pair_ratio"]
                    for k in range(1, s["tgs_per_core"] + 1):
                        P = NUM_GPU_CORES * k
                        if k > 1 and P * csg > total_tiles:
                        	break
                        rounds = _ceil_div(total_tiles, P * csg)
                        n_c = k * csg
                        # est_us = rounds * mma_tiles * max(C_SG, n_c * C_ALU) / GPU_CLOCK * 1e6
                        est_us = rounds * mma_tiles * C_SG_CAL[(rm, rn)] / GPU_CLOCK * 1e6
                        cands.append(dict(
                            block_m=bm, block_n=bn, block_k=BLOCK_K,
                            reg_tile_m=rm, reg_tile_n=rn,
                            compute_sg=csg, write_sg=s["write_sg"], ring_buf=RING_BUF,
                            group_k=group_k, num_m_blocks=num_m_blocks, num_n_blocks=num_n_blocks,
                            persist_tgs=P, threads_per_tg=s["threads"],
                            est_us=est_us, n_c=n_c, rounds=rounds))

    if not cands:
    	return None
    best_est = min(c["est_us"] for c in cands)
    near = [c for c in cands if c["est_us"] <= best_est * 1.05]
    near.sort(key=lambda c: (0 if PREFERRED.get((M, N)) == (c["block_m"], c["block_n"]) else 1, c["est_us"]))
    if True:
    	for c in sorted(cands, key=lambda c: c["est_us"])[:5]:
    		print("cand", c["block_m"], c["block_n"], "P", c["persist_tgs"],
    			"rounds", c["rounds"], "est_us", round(c["est_us"]))
    return near[0]

def get_fp16_gemm_config(M, N, K, cd_dtype=np.float32, with_accum=False, verbose=True):
	elem_cd = np.dtype(cd_dtype).itemsize
	best = _search(M, N, K, elem_cd, with_accum, allow_ragged=False, verbose=verbose)

	# if best is None:
	# 	best = _search(M, N, K, elem_cd, with_accum, allow_ragged=True, verbose=verbose)

	if best is None:
		raise ValueError(f"GEMM {M}, {N}, {K}在当前候选下没有可行配置")
	if verbose:
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
