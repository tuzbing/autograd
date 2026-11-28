#!/usr/bin/env python3
"""
python download_dataset.py --output=~/Documents/datasets/p19 --workers 10
默认目标：https://huggingface.co/datasets/emozilla/pg19/tree/main/data
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PRINT_LOCK = threading.Lock()
CHUNK_SIZE = 1024 * 1024  # 1 MB


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """跨域重定向时剥离 Authorization 请求头，避免 S3/CDN 返回 400 错误。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        old_host = urllib.parse.urlparse(req.full_url).netloc
        new_host = urllib.parse.urlparse(new_req.full_url).netloc
        if old_host != new_host:
            new_req.remove_header("Authorization")
        return new_req


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def make_headers(token: str | None = None, extra: dict | None = None) -> dict:
    headers = {"User-Agent": "python-stdlib-hf-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def list_dataset_files(
    repo_id: str,
    revision: str,
    subpath: str,
    endpoint: str,
    token: str | None = None,
    timeout: int = 30,
) -> list[dict]:
    """调用 Hugging Face Tree API 列出目录下所有文件（支持自动翻页）。"""
    subpath = subpath.strip("/")
    base = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}/tree/{revision}"
    url = f"{base}/{urllib.parse.quote(subpath, safe='/')}?recursive=true" if subpath else f"{base}?recursive=true"

    opener = urllib.request.build_opener(SafeRedirectHandler())
    files: list[dict] = []

    while url:
        req = urllib.request.Request(url, headers=make_headers(token))
        with opener.open(req, timeout=timeout) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            for item in items:
                if item.get("type") == "file":
                    lfs = item.get("lfs") or {}
                    files.append(
                        {
                            "path": item["path"],
                            "size": lfs.get("size") or item.get("size"),
                            "sha256": lfs.get("oid"),
                        }
                    )

            # 检查是否存在 Link 分页头
            link_header = resp.headers.get("Link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            url = match.group(1) if match else None

    return files


def calc_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    file_info: dict,
    repo_id: str,
    revision: str,
    endpoint: str,
    output_dir: Path,
    token: str | None,
    verify_hash: bool,
    max_retries: int,
    timeout: int,
) -> tuple[str, bool, str]:
    """下载单个文件，支持断点续传、失败重试与 SHA-256 校验。"""
    rel_path = file_info["path"]
    expected_size = file_info["size"]
    expected_sha256 = file_info["sha256"]

    dest_path = output_dir / rel_path
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 已存在且大小一致则跳过
    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if expected_size is None or local_size == expected_size:
            if not verify_hash or not expected_sha256 or calc_sha256(dest_path) == expected_sha256:
                log(f"[跳过] {rel_path} ({format_size(local_size)})")
                return rel_path, True, "skipped"
            log(f"[校验失败，重新下载] {rel_path}")
            dest_path.unlink()

    encoded_path = urllib.parse.quote(rel_path, safe="/")
    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{encoded_path}"
    opener = urllib.request.build_opener(SafeRedirectHandler())

    for attempt in range(1, max_retries + 1):
        try:
            existing = part_path.stat().st_size if part_path.exists() else 0

            # 如果 .part 文件异常大于目标大小，重新开始
            if expected_size is not None and existing > expected_size:
                part_path.unlink()
                existing = 0

            # 若 .part 已完整，直接进入校验与重命名
            if expected_size is None or existing < expected_size:
                extra_headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
                req = urllib.request.Request(url, headers=make_headers(token, extra_headers))

                if existing > 0:
                    log(f"[续传] {rel_path} 从 {format_size(existing)} 继续 (尝试 {attempt}/{max_retries})")
                else:
                    log(f"[开始] {rel_path} ({format_size(expected_size or 0)})")

                with opener.open(req, timeout=timeout) as resp:
                    status = resp.status
                    if existing > 0 and status == 206:
                        mode = "ab"
                        downloaded = existing
                    else:
                        mode = "wb"
                        downloaded = 0

                    last_report = time.monotonic()
                    with part_path.open(mode) as f:
                        while True:
                            chunk = resp.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)

                            now = time.monotonic()
                            if now - last_report >= 10:
                                pct = f"{downloaded / expected_size * 100:.1f}%" if expected_size else "?"
                                log(f"[进度] {rel_path}: {format_size(downloaded)} / {format_size(expected_size or 0)} ({pct})")
                                last_report = now

            # 校验文件大小
            final_size = part_path.stat().st_size
            if expected_size is not None and final_size != expected_size:
                raise IOError(f"文件大小不匹配: 实际 {final_size}, 期望 {expected_size}")

            # 校验 LFS SHA-256
            if verify_hash and expected_sha256:
                actual_sha256 = calc_sha256(part_path)
                if actual_sha256 != expected_sha256:
                    part_path.unlink(missing_ok=True)
                    raise IOError(f"SHA-256 校验失败: 实际 {actual_sha256[:12]}, 期望 {expected_sha256[:12]}")

            part_path.replace(dest_path)
            log(f"[完成] {rel_path} ({format_size(final_size)})")
            return rel_path, True, "downloaded"

        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return rel_path, False, f"HTTP {e.code}: {e.reason}"
            wait = min(2**attempt, 30)
            log(f"[重试] {rel_path} 遇到 HTTP {e.code}，{wait}s 后重试 ({attempt}/{max_retries})")
            time.sleep(wait)
        except Exception as e:
            wait = min(2**attempt, 30)
            log(f"[重试] {rel_path} 出错 ({e})，{wait}s 后重试 ({attempt}/{max_retries})")
            time.sleep(wait)

    return rel_path, False, f"超过最大重试次数 ({max_retries})"


def main() -> None:
    parser = argparse.ArgumentParser(description="零依赖下载 Hugging Face 数据集目录")
    parser.add_argument("--repo", default="emozilla/pg19", help="数据集仓库 ID")
    parser.add_argument("--revision", default="main", help="分支或 commit")
    parser.add_argument("--subpath", default="data", help="仓库内目标子目录")
    parser.add_argument("-o", "--output", default="./pg19_dataset", help="本地保存目录")
    parser.add_argument("-w", "--workers", type=int, default=3, help="并发下载线程数（建议 2-4）")
    parser.add_argument("--retries", type=int, default=5, help="单文件最大重试次数")
    parser.add_argument("--timeout", type=int, default=60, help="网络超时秒数")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face 端点（如使用镜像可设为 https://hf-mirror.com）",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="可选的 Hugging Face Access Token（也可通过环境变量 HF_TOKEN 传入）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过 LFS 文件的 SHA-256 校验")
    args = parser.parse_args()

    # output_dir = Path(args.output).resolve()
    output_dir = Path(args.output).expanduser().resolve()
    log(f"正在获取文件列表: {args.endpoint}/datasets/{args.repo}/tree/{args.revision}/{args.subpath}")

    try:
        files = list_dataset_files(
            repo_id=args.repo,
            revision=args.revision,
            subpath=args.subpath,
            endpoint=args.endpoint,
            token=args.token,
            timeout=args.timeout,
        )
    except Exception as e:
        log(f"获取文件列表失败: {e}")
        sys.exit(1)

    if not files:
        log("未找到任何文件，请检查仓库名或目录路径。")
        sys.exit(1)

    total_size = sum(f["size"] or 0 for f in files)
    log(f"共发现 {len(files)} 个文件，总大小约 {format_size(total_size)}，保存目录: {output_dir}\n")

    failed: list[tuple[str, str]] = []
    start_time = time.monotonic()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    download_one,
                    f,
                    args.repo,
                    args.revision,
                    args.endpoint,
                    output_dir,
                    args.token,
                    not args.no_verify,
                    args.retries,
                    args.timeout,
                )
                for f in files
            ]
            for fut in concurrent.futures.as_completed(futures):
                rel_path, ok, detail = fut.result()
                if not ok:
                    failed.append((rel_path, detail))
    except KeyboardInterrupt:
        log("\n用户中断下载。未完成的文件已保留为 .part，重新运行脚本将自动断点续传。")
        sys.exit(130)

    elapsed = time.monotonic() - start_time
    log(f"\n处理结束，耗时 {elapsed:.1f}s。成功: {len(files) - len(failed)}/{len(files)}")

    if failed:
        log("以下文件下载失败（可直接重新运行脚本续传）：")
        for path, reason in failed:
            log(f"  - {path}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()