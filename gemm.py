
import numpy as np
import math

# threadgroup memory
TG_CAPACITY = 32768 # 32KB

SCALE_GROUP = 128
BLOCK_K = 32
MAX_STAGES = 16
MMA_M = 8
BLOCK_N_MAX = 128

TG_ALIGN = 16
SYNC_BYTES = 256
ATOMIC_UINT_BYTES = 4
MMA_SCRATCH_TILE_BYTES = 8 * 8 * 4
OTHER_BYTES = 0

NUM_GPU_CORES = 8
MEMORY_BANDWIDTH_BYTES_PER_SECOND = 68.25e9
GPU_CLOCK_CYCLES_PER_SECOND = 1.296e9
MEMORY_BYTES_PER_CYCLE = MEMORY_BANDWIDTH_BYTES_PER_SECOND / GPU_CLOCK_CYCLES_PER_SECOND

MMA_CYCLES_PER_OP = 1.0
REG_LOAD_CYCLES_PER_STEP = 1.0

MAX_RESIDENT_THREADS_PER_CORE = 1.0 # 每个core允许的最大常驻线程数
MAX_TGS_PER_CORE_HW_LIMIT = 1.0 # 硬件的threadgroup调度槽位数限制

SYNC_LATENCY_CYCLES = 2

ESTIMATED_REGS_PER_THREAD = {
    (64, 48, 1): 160,
    (64, 48, 2): 160,
    (128, 96, 1): 208,
}


def _ceil_div(a, b):
	return (a + b - 1) // b

def _align(n, alignment=TG_ALIGN):
	return _ceil_div(n, alignment) * alignment

def max_resident_threads_by_regs(regs):
    # 对应 Alyssa 文中的阶梯表（Apple7，M1同代架构）
    table = [(104,1024),(112,896),(128,832),(144,704),
             (160,640),(184,576),(208,512),(232,448),(256,384)]
    for reg_limit, threads in table:
        if regs <= reg_limit:
            return threads
    return 256  # 超出表范围时的保守兜底值，需要用spill实测修正

def compute_num_threads(block_m, block_n, loader_sg=1):
    num_math_simdgroups = 4 if block_m*block_n <= 64*64 else 16
    return (num_math_simdgroups + loader_sg) * 32

def estimate_regs_per_thread_fallback(block_m, block_n):
    num_math_simdgroups = 4 if block_m*block_n <= 64*64 else 16
    num_fragments = _ceil_div(block_m, 8) * _ceil_div(block_n, 8)
    fragments_per_sg = _ceil_div(num_fragments, num_math_simdgroups)
    base = 56
    per_fragment = 4
    return base + fragments_per_sg * per_fragment

def compute_occupancy(threadgroup_bytes, num_threads, max_persist_threads_per_core):
	tgs_by_smem = TG_CAPACITY // threadgroup_bytes
	tgs_by_threads = max_persist_threads_per_core // num_threads
	tgs_per_core = max(1, min(tgs_by_smem, tgs_by_threads))
	return tgs_per_core

def is_reg_spill(block_m, block_n, num_math_simdgroups):
	f = (block_m * block_n) / (64 * num_math_simdgroups)
	return max(0, 8*(f-4))

def get_fp16_gemm_config(M, N, K, cd_dtype=np.float32, with_accum=False):
	cd_dtype = np.dtype(cd_dtype)
	elem_cd = cd_dtype.itemsize
	elem_ab = 2
	block_ms = [16, 32, 64, 128]
	block_ns = range(16, BLOCK_N_MAX+1, 16)

	group_k = _ceil_div(K, SCALE_GROUP)
	best = None

	for block_m in block_ms:
		for block_n in block_ns:
			use_uniform_sb = 1 if BLOCK_K % block_n == 0 else 2
			num_math_simdgroups = 4 if block_m*block_n <= 64*64 else 16

			# 固定区: C/D暂存区 + 当前输出tile所需的B scale区域 + 同步/其他状态
			smem_cd = 0
			smem_sfb = _align(group_k*4*use_uniform_sb)
			# fixed_bytes = smem_cd + smem_sfb + _align(SYNC_BYTES)
			smem_mma_scratch = num_math_simdgroups * MMA_SCRATCH_TILE_BYTES

			# 单stage: A + B + ScaleA
			smem_a = _align(block_m * BLOCK_K * elem_ab)
			smem_b = _align(block_n * BLOCK_K * elem_ab)
			smem_sfa = _align(block_m * 4)
			stage_bytes = smem_a + smem_b + smem_sfa

			# available = TG_CAPACITY - fixed_bytes
			# if available < stage_bytes:
			# 	continue

			# max_num_stages = min(MAX_STAGES, available // stage_bytes)

			# 寄存器是否溢出
			reg_splill_bytes = is_reg_spill(block_m, block_n, num_math_simdgroups)
			if reg_splill_bytes > 0.0:
				print(f"block_m {block_m}, block_n {block_n}, register spill bytes {reg_splill_bytes}")
				continue

			# 过滤掉num_stages<2
			for num_stages in range(2, MAX_STAGES+1):
				smem_sync = _align((3 * num_stages + 4) * ATOMIC_UINT_BYTES)
				fixed_bytes = smem_cd + smem_sfb + smem_sync + _align(smem_mma_scratch)
				threadgroup_bytes = fixed_bytes + num_stages * stage_bytes
				if threadgroup_bytes > TG_CAPACITY:
					print(f"block_m {block_m}, block_n {block_n} threadgroup_bytes > 32KB skip")
					continue
				num_threads = compute_num_threads(block_m, block_n, loader_sg=1)
				regs_per_thread = ESTIMATED_REGS_PER_THREAD.get((block_m, block_n, num_stages), estimate_regs_per_thread_fallback(block_m, block_n))
				max_persist_threads_per_core = max_resident_threads_by_regs(regs_per_thread)
				tgs_per_core = compute_occupancy(threadgroup_bytes, num_threads, max_persist_threads_per_core)

				num_m_blocks = _ceil_div(M, block_m)
				num_n_blocks = _ceil_div(N, block_n)
				num_blocks = num_m_blocks * num_n_blocks

				max_persist_tgs = NUM_GPU_CORES * tgs_per_core
				persist_tgs = min(max_persist_tgs, num_blocks)
				P = persist_tgs   # 这个P要传给dispatch，而不是写死8
				num_waves = _ceil_div(num_blocks, P)
				# wave_efficiency = num_blocks / (num_waves * P)

				# 计算周期
				num_k_tiles = _ceil_div(K, BLOCK_K)
				num_fragments = _ceil_div(block_m, 8) * _ceil_div(block_n, 8)
				fragments_per_sg = _ceil_div(num_fragments, num_math_simdgroups)
				mma_per_sg = fragments_per_sg * (BLOCK_K // 8)
				mma_cycles_per_block = num_k_tiles * mma_per_sg * MMA_CYCLES_PER_OP
				shared_to_reg_cycles_per_block = num_k_tiles * mma_per_sg * REG_LOAD_CYCLES_PER_STEP
				compute_cycles = math.ceil((mma_cycles_per_block + shared_to_reg_cycles_per_block) * num_waves)

				# 同步周期
				handshakes_per_tg = num_waves * num_k_tiles
				pipeline_overlap = 1 if num_stages >= 2 else 0
				hiding_lanes = max(1, tgs_per_core) + pipeline_overlap
				sync_cycles = math.ceil(handshakes_per_tg * SYNC_LATENCY_CYCLES / hiding_lanes)

				# 数据搬运周期
				bytes_ab = K * (block_m + block_n) * elem_ab
				bytes_sfa = block_m * group_k * 4 # scale A
				bytes_sfb = group_k * use_uniform_sb * 4 # scale B
				bytes_cd = block_m * block_n * elem_cd * (2 if with_accum else 1)
				bytes_per_block = bytes_ab + bytes_sfa + bytes_sfb + bytes_cd
				memory_cycles = math.ceil((bytes_per_block * num_blocks) / MEMORY_BYTES_PER_CYCLE)

				if num_stages <= 1 and tgs_per_core <= 1:
					num_cycles = memory_cycles + compute_cycles
				else:
					num_cycles = max(memory_cycles, compute_cycles)

				# num_cycles += sync_cycles

				print(f"M={M}, N={N}, K={K}, block_m {block_m}, block_n {block_n},"
					  f" tgs_per_core {tgs_per_core}, P {P}, sync_cycles {sync_cycles}, "
					  f" compute_cycles {compute_cycles}, memory_cycles {memory_cycles} num_stages {num_stages}")

				if best is not None and num_cycles >= best["num_cycles"]:
					continue

				best = {
					"block_m": block_m,
					"block_n": block_n,
					"block_k": BLOCK_K,
					"group_k": group_k,
					"num_stages": num_stages,
					"threadgroup_bytes": threadgroup_bytes,
					"num_m_blocks": num_m_blocks,
					"num_n_blocks": num_n_blocks,
					"num_cycles": num_cycles,
					"persist_tgs": persist_tgs,
				}

	if best is None:
		raise ValueError(f"GEMM {M}, {N}, {K}在当前候选和共享内存布局下无法容纳一个stage")

	return best

def fp16_gemm_kernel_name(cfg, with_accum):
	return (f"fp16_gemm_"
		f"bm{int(cfg['block_m'])}_"
		f"bn{int(cfg['block_n'])}_"
		f"bk{int(cfg['block_k'])}_"
		f"gk{int(cfg['group_k'])}_"
		f"s{int(cfg['num_stages'])}_"
		f"a{int(bool(with_accum))}"
	)















