
import numpy as np
import math

# threadgroup memory
TG_CAPACITY = 32768 # 32KB
SIMD_WIDTH = 32
MAX_SG_PER_TG = 32 # 1024线程/32
NUM_GPU_CORES = 8

SCALE_GROUP = 128
FRAG = 8

TG_ALIGN = 16
ATOMIC_UINT_BYTES = 4
RAW_ACCUM_BYTES = FRAG*FRAG*4
SLOT_META_BYTES = 16
SLOT_BYTES = RAW_ACCUM_BYTES + SLOT_META_BYTES

MEMORY_BANDWIDTH_BYTES_PER_SECOND = 68.25e9
GPU_CLOCK_CYCLES_PER_SECOND = 1.296e9
MEMORY_BYTES_PER_CYCLE = MEMORY_BANDWIDTH_BYTES_PER_SECOND / GPU_CLOCK_CYCLES_PER_SECOND

WRITE_CYCLES_PER_MESSAGE = 1.0
MMA_CYCLES_PER_OP = 1.0

REG_BASE = 64        # 固定开销（寻址/循环变量等），粗估值，待校正
REG_PER_SLOT = 2      # 每个"寄存器驻留槽位"（累加器或临时fragment）均摊的寄存器数，粗估值，待校正
REG_TILE_CANDIDATS = (1, 2, 4, 8)

# MIN_TOTAL_SG_FRACTION = 0.5  # 阈值待校准,先给一个"至少用到一半的32上限"的保守起点
T_LOAD_US = 0.14
T_MSG_US = 1.0
# REQ_BW_BYTES_PER_US=78e3
REQ_BW_BYTES_PER_US = 64e3
FRAG_BYTES = 128

MAX_WRITE_REG_SLOTS = 16          # pair_ratio * rm * rn 上限（write 端 acc 寄存器）
PAIR_RATIO_CANDIDATES = (1, 2, 4)
RING_BUF_CANDIDATES = (1, 2, 4)

T_MSG_BASE_US = 1.0               # 占位：一条消息的固定开销（轮询、meta、scale load 往返），待标定
T_MSG_FRAG_US = 0.2               # 占位：每个 fragment 的额外开销，待标定


def _ceil_div(a, b):
	return (a + b - 1) // b

def _align(n, alignment=TG_ALIGN):
	return _ceil_div(n, alignment) * alignment

def slot_bytes(rm, rn):
    return rm * rn * 256 + 16     # 与 kernel 的 SLOT_BYTES 一致

def smem_bytes(compute_sg, ring_buf, rm, rn):
    return _align(compute_sg * ring_buf * (slot_bytes(rm, rn) + ATOMIC_UINT_BYTES))

def _full_subs(extent, block, rfrags_elems, subs):
    # 镜像 messages_for_tile 在一个维度上的求和：整块贡献 subs，余块贡献 min(rem // (R*8), subs)
    return (extent // block) * subs + min((extent % block) // rfrags_elems, subs)

def count_messages(M, N, block_m, block_n, rm, rn, group_k):
    subs_m = block_m // FRAG // rm
    subs_n = block_n // FRAG // rn
    nmb = _ceil_div(M, block_m)
    nnb = _ceil_div(N, block_n)
    interior = _full_subs(M, block_m, rm * FRAG, subs_m) * _full_subs(N, block_n, rn * FRAG, subs_n)
    total_subs = nmb * nnb * subs_m * subs_n
    # interior 子块每组 1 条消息，非 interior 子块每个 fragment 每组 1 条
    return group_k * (interior + (total_subs - interior) * rm * rn)

def max_resident_threads_by_regs(regs):
    # 对应 Alyssa 文中的阶梯表（Apple7，M1同代架构）
    table = [(104,1024),(112,896),(128,832),(144,704),
             (160,640),(184,576),(208,512),(232,448),(256,384)]
    for reg_limit, threads in table:
        if regs <= reg_limit:
            return threads
    return 256  # 超出表范围时的保守兜底值，需要用spill实测修正

def estimate_regs_by_thread(reg_tile_m, reg_tile_n, pair_ratio):
    """
    粗粒度估算：compute 路径的槽位数 = RM*RN（累加器）+ RM+RN（临时fragment），
    write 路径的槽位数 = PAIR_RATIO（persistent_accum）。
    两条路径共用一个编译单元，寄存器分配取较大值。
    这是一阶近似，用同一个 REG_PER_SLOT 权重，不再区分累加器/临时寄存器/write累加器，
    后续用实测反汇编结果拆分成更精确的分项权重。
    """
    compute_slots = reg_tile_m * reg_tile_n + reg_tile_m + reg_tile_n
    write_slots = 2 * pair_ratio * reg_tile_m * reg_tile_n
    regs_compute = REG_BASE + REG_PER_SLOT * compute_slots
    regs_write = REG_BASE + REG_PER_SLOT * write_slots
    return max(regs_compute, regs_write)

# compute_cycles_per_tile ∝ NUM_FRAGS * num_k_tiles * MMA_CYCLES_PER_OP
# write_cycles_per_tile   ∝ NUM_FRAGS * group_k      * WRITE_CYCLES_PER_MESSAGE
# compute_total = total_tiles/(P*COMPUTE_SG_NUM) * compute_cycles_per_tile
# write_total   = total_tiles/(P*WRITE_SG_NUM)   * write_cycles_per_tile
def calibrate_pair_ratio(block_k):
    """
    compute_total / write_total = (SCALE_GROUP/block_k) * (MMA_CYCLES_PER_OP/WRITE_CYCLES_PER_MESSAGE)
    候选集直接取 SCALE_GROUP/block_k 在 block_k∈{8,16,32,64,128} 下的精确值 {16,8,4,2,1}。
    """
    candidates = (1, 2, 4, 8, 16)
    lhs = (SCALE_GROUP / block_k) * (MMA_CYCLES_PER_OP / WRITE_CYCLES_PER_MESSAGE)
    return min(candidates, key=lambda r: abs(lhs - r))

def effective_ab_bytes(block_m, block_n, K, elem_ab, reg_tile_m, reg_tile_n):
    """
    简化版：不假设任何 L1/L2 容量（M1 无公开数据，猜测阈值意义不大），
    统一按"最坏情况"处理 —— 冗余读取倍数 = NUM_FRAGS / REG_TILE，恒定生效，
    不给任何缓存命中折扣。保留了"RM/RN越大，重读倍数越小"这个核心修正方向，
    后续若发现模型系统性高估/低估访存量，再引入一个全局标定系数或重新考虑缓存建模。
    """
    num_m_frags = block_m // FRAG
    num_n_frags = block_n // FRAG

    amp_a = max(1, num_n_frags // reg_tile_n)
    amp_b = max(1, num_m_frags // reg_tile_m)

    bytes_a = amp_a * block_m * K * elem_ab
    bytes_b = amp_b * block_n * K * elem_ab
    return bytes_a + bytes_b

def is_valid_pipeline_shape(block_k, compute_sg, write_sg):
    """只检查与具体 block_m/block_n 无关的流水线组合是否合法。"""
    if block_k % FRAG != 0:
        return False
    if SCALE_GROUP % block_k != 0:
        return False
    if compute_sg < write_sg or compute_sg % write_sg != 0:
        return False
    return True

def is_valid_block_tile_params(block_m, block_n, reg_tile_m, reg_tile_n):
    """检查具体 block_m/block_n 与 reg_tile 的整除关系，对应 kernel 里新增的 static_assert。
    必须用真实的 block_m/block_n 调用，不能用占位符。"""
    if block_m % FRAG != 0 or block_n % FRAG != 0:
        return False
    if reg_tile_m < 1 or reg_tile_n < 1:
        return False
    num_m_frags = block_m // FRAG
    num_n_frags = block_n // FRAG
    if num_m_frags % reg_tile_m != 0:
        return False
    if num_n_frags % reg_tile_n != 0:
        return False
    return True

def get_pipeline_shapes(block_k, reg_tile_m, reg_tile_n):
	shapes = []
	sub_frags = reg_tile_m * reg_tile_n
	for pair_ratio in PAIR_RATIO_CANDIDATES:
		if pair_ratio * sub_frags > MAX_WRITE_REG_SLOTS:
			continue
		for write_sg in range(1, MAX_SG_PER_TG // (pair_ratio + 1) + 1):
			compute_sg = pair_ratio * write_sg
			if compute_sg + write_sg > MAX_SG_PER_TG:
				continue
			if not is_valid_pipeline_shape(block_k, compute_sg, write_sg):
				continue
			for ring_buf in RING_BUF_CANDIDATES:
				smem = smem_bytes(compute_sg, ring_buf, reg_tile_m, reg_tile_n)
				if smem > TG_CAPACITY:
					continue
				shapes.append({
					"pair_ratio": pair_ratio,
					"compute_sg": compute_sg,
					"write_sg": write_sg,
					"ring_buf": ring_buf,
					"threadgroup_bytes": smem,
					})
	return shapes

def get_fp16_gemm_config(M, N, K, cd_dtype=np.float32, with_accum=False):
	cd_dtype = np.dtype(cd_dtype)
	elem_cd = cd_dtype.itemsize
	elem_ab = 2

	group_k = _ceil_div(K, SCALE_GROUP)
	best = None

	for block_k in (128, 64, 32, 16, 8):
		if SCALE_GROUP % block_k != 0:
			continue
		num_k_tiles = _ceil_div(K, block_k)

		for reg_tile_m in REG_TILE_CANDIDATS:
			for reg_tile_n in REG_TILE_CANDIDATS:
				shapes = get_pipeline_shapes(block_k, reg_tile_m, reg_tile_n)
				if not shapes:
					continue

				for shape in shapes:
					compute_sg = shape["compute_sg"]
					write_sg = shape["write_sg"]
					pair_ratio = shape["pair_ratio"]
					threadgroup_bytes = shape["threadgroup_bytes"]
					ring_buf = shape["ring_buf"]
					num_threads = (compute_sg + write_sg) * SIMD_WIDTH

					regs_per_thread = estimate_regs_by_thread(reg_tile_m, reg_tile_n, pair_ratio)
					max_persist_threads = max_resident_threads_by_regs(regs_per_thread)
					tgs_by_smem = max(1, TG_CAPACITY // threadgroup_bytes)
					tgs_by_threads = max(1, max_persist_threads // num_threads)
					tgs_per_core = max(1, min(tgs_by_smem, tgs_by_threads))
					max_persist_tgs = NUM_GPU_CORES * tgs_per_core

					for block_m in (8, 16, 32, 64, 128):
						num_m_frags = block_m // FRAG
						if reg_tile_m > num_m_frags:
							continue

						for block_n in range(8, 257, 8):
							num_n_frags = block_n // FRAG
							if reg_tile_n > num_n_frags:
								continue
							if not is_valid_block_tile_params(block_m, block_n, reg_tile_m, reg_tile_n):
								continue
							num_m_blocks = _ceil_div(M, block_m)
							num_n_blocks = _ceil_div(N, block_n)
							total_tiles = num_m_blocks * num_n_blocks

							P = min(max_persist_tgs, total_tiles)
							num_frags = num_m_frags * num_n_frags
							active_c = min(total_tiles, P * compute_sg, 96)
							active_w = min(P*write_sg, _ceil_div(active_c, pair_ratio))

							bytes_ab_tile = effective_ab_bytes(block_m, block_n, K, elem_ab, reg_tile_m, reg_tile_n)
							loads_per_tile = bytes_ab_tile / FRAG_BYTES
							tiles_per_sg = _ceil_div(total_tiles, active_c)      # 含负载不均
							t_load_us = loads_per_tile * tiles_per_sg * T_LOAD_US

							bytes_sfa = block_m * group_k * 4
							bytes_sfb = block_n * group_k * 4
							bytes_cd = block_m * block_n * elem_cd * (2 if with_accum else 1)
							total_bytes = (bytes_ab_tile + bytes_sfa + bytes_sfb + bytes_cd) * total_tiles
							t_bw_us = total_bytes / REQ_BW_BYTES_PER_US

							total_msgs = count_messages(M, N, block_m, block_n, reg_tile_m, reg_tile_n, group_k)
							t_msg_us = T_MSG_BASE_US + T_MSG_FRAG_US * reg_tile_m * reg_tile_n
							t_write_us = total_msgs / active_w * t_msg_us

							est_us = max(t_load_us, t_bw_us, t_write_us)

							print(f"M={M}, N={N}, K={K}, block_m {block_m}, block_n {block_n}, block_k {block_k}, "
							  f" RM {reg_tile_m}, RN {reg_tile_n}, tgs_per_core {tgs_per_core}, P {P}, "
							  f" compute_sg_num {compute_sg}, write_sg_num {write_sg}, ring_buf {ring_buf}"
							  f" t_load_us {t_load_us}, t_bw_us {t_bw_us}, t_write_us {t_write_us}")

							if best is not None and est_us >= best["est_us"]:
								continue

							best = dict(block_m=block_m, block_n=block_n, block_k=block_k,
										reg_tile_m=reg_tile_m, reg_tile_n=reg_tile_n,
										compute_sg=compute_sg, write_sg=write_sg, ring_buf=shape["ring_buf"],
										group_k=group_k, num_m_blocks=num_m_blocks, num_n_blocks=num_n_blocks,
										persist_tgs=P, est_us=est_us)

	if best is None:
		raise ValueError(f"GEMM {M}, {N}, {K}在当前候选和共享内存布局下无法容纳一个stage")

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















