"""验证 FTU 小范围导出结果是否正确且合理。

用途：
1) 校验 exports/FTU 的目录结构、manifest、meta、npz 之间是否一致
2) 在可用 numpy 时，增加时间轴/HR/RR/雷达数值合理性检查
3) 为后续大规模导出提供快速验收依据

示例：
python src/valid/validate_ftu_export.py
python src/valid/validate_ftu_export.py --expected-count 4 --expected-distance "40 cm"
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORTS_FTU = ROOT / "exports" / "FTU"


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    sample_tag: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FTU export artifacts.")
    parser.add_argument(
        "--exports-ftu",
        type=Path,
        default=DEFAULT_EXPORTS_FTU,
        help="FTU 导出目录（默认: ./exports/FTU）",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=4,
        help="期望样本数量（默认 4，适配当前小范围导出）",
    )
    parser.add_argument(
        "--expected-participant",
        type=int,
        default=1,
        help="期望 participant_id（默认 1）",
    )
    parser.add_argument(
        "--expected-scenario",
        type=str,
        default="Distance",
        help="期望 scenario（默认 Distance）",
    )
    parser.add_argument(
        "--expected-distance",
        type=str,
        default="40 cm",
        help="期望 distance（默认 40 cm）",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default="/Radar_Vital_Signs/tmp/ftu_validation_report.json",
        help="可选：输出 JSON 报告路径",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_manifest_shape(shape_str: str) -> Optional[Tuple[int, ...]]:
    try:
        parsed = ast.literal_eval(shape_str)
        if isinstance(parsed, tuple) and all(isinstance(x, int) for x in parsed):
            return parsed
    except Exception:
        return None
    return None


def read_npy_header_from_bytes(data: bytes) -> Optional[Dict[str, Any]]:
    """仅读取 .npy 头，不依赖 numpy。"""
    if data[:6] != b"\x93NUMPY":
        return None
    major = data[6]
    if major == 1:
        hlen = struct.unpack("<H", data[8:10])[0]
        start = 10
    elif major in (2, 3):
        hlen = struct.unpack("<I", data[8:12])[0]
        start = 12
    else:
        return None

    header_text = data[start : start + hlen].decode("latin1")
    try:
        return ast.literal_eval(header_text)
    except Exception:
        return None


def read_npz_headers(npz_path: Path) -> Dict[str, Dict[str, Any]]:
    headers: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(npz_path, "r") as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            header = read_npy_header_from_bytes(zf.read(name))
            if header is None:
                raise ValueError(f"无法解析 nparray 头信息: {npz_path.name}:{name}")
            headers[key] = header
    return headers


def resolve_npz_path(manifest_npz_path: str, exports_ftu: Path, sample_tag: str) -> Path:
    # manifest 中路径可能来自其它容器工作目录，优先兜底到当前 exports
    candidate = Path(manifest_npz_path)
    if candidate.exists():
        return candidate
    fallback = exports_ftu / "samples" / f"{sample_tag}.npz"
    return fallback


def maybe_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def check_numeric_reasonableness(np: Any, npz_path: Path, sample_tag: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    data = np.load(npz_path)

    required = {"radar", "time", "heart_rate", "respiration_rate"}
    keys = set(data.files)
    if keys != required:
        issues.append(ValidationIssue("error", sample_tag, f"npz keys 不匹配: {sorted(keys)}"))
        return issues

    radar = data["radar"]
    time = data["time"]
    heart_rate = data["heart_rate"]
    respiration_rate = data["respiration_rate"]

    if radar.ndim != 4:
        issues.append(ValidationIssue("error", sample_tag, f"radar 维度应为4，实际 {radar.ndim}"))
    if radar.shape[1:] != (8, 16, 8):
        issues.append(ValidationIssue("error", sample_tag, f"radar 非目标 RDA 尺寸: {radar.shape}"))
    if not np.iscomplexobj(radar):
        issues.append(ValidationIssue("error", sample_tag, "radar 应为复数张量"))

    # 有限值检查
    radar_finite = np.isfinite(radar.real).all() and np.isfinite(radar.imag).all()
    if not radar_finite:
        issues.append(ValidationIssue("error", sample_tag, "radar 存在非有限值 (NaN/Inf)"))

    # 时间轴：长度一致，非降序
    n = len(time)
    if len(heart_rate) != n or len(respiration_rate) != n:
        issues.append(
            ValidationIssue(
                "error",
                sample_tag,
                f"time/hr/rr 长度不一致: {len(time)}/{len(heart_rate)}/{len(respiration_rate)}",
            )
        )
    if n > 1 and np.any(np.diff(time) < 0):
        issues.append(ValidationIssue("error", sample_tag, "time 非单调非降序"))

    # HR 物理合理性（仅检查有限值）
    hr_finite = heart_rate[np.isfinite(heart_rate)]
    if hr_finite.size == 0:
        issues.append(ValidationIssue("error", sample_tag, "heart_rate 全为 NaN/Inf"))
    else:
        hr_min, hr_max = float(hr_finite.min()), float(hr_finite.max())
        if hr_min < 25 or hr_max > 240:
            issues.append(
                ValidationIssue(
                    "warning",
                    sample_tag,
                    f"heart_rate 超出常见范围: min={hr_min:.2f}, max={hr_max:.2f}",
                )
            )

    # FTU 当前没有 RR，导出应主要为空值
    rr_finite_ratio = float(np.isfinite(respiration_rate).mean())
    if rr_finite_ratio > 0.2:
        issues.append(
            ValidationIssue(
                "warning",
                sample_tag,
                f"respiration_rate 有效值比例偏高({rr_finite_ratio:.2%})，与 FTU 当前预期不符",
            )
        )

    return issues


def main() -> None:
    args = parse_args()
    exports_ftu = args.exports_ftu.resolve()
    issues: List[ValidationIssue] = []

    manifest_path = exports_ftu / "manifest.csv"
    meta_dir = exports_ftu / "meta"
    samples_dir = exports_ftu / "samples"

    if not exports_ftu.exists():
        print(f"[ERROR] 导出目录不存在: {exports_ftu}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"[ERROR] 缺少 manifest: {manifest_path}")
        sys.exit(1)
    if not meta_dir.exists() or not samples_dir.exists():
        print(f"[ERROR] 缺少 meta/samples 子目录: {exports_ftu}")
        sys.exit(1)

    rows = read_csv_rows(manifest_path)
    if not rows:
        print("[ERROR] manifest 为空")
        sys.exit(1)

    # 全局预期（适配当前小范围导出）
    if args.expected_count is not None and len(rows) != args.expected_count:
        issues.append(
            ValidationIssue(
                "warning",
                "GLOBAL",
                f"样本数 {len(rows)} 与 expected_count={args.expected_count} 不一致",
            )
        )

    for row in rows:
        sample_tag = row.get("sample_tag", "")
        status = row.get("status", "")
        if status != "ok":
            issues.append(ValidationIssue("error", sample_tag, f"manifest status 非 ok: {status}"))
            continue

        # 小范围导出预期字段
        if int(row.get("participant_id", -1)) != args.expected_participant:
            issues.append(ValidationIssue("warning", sample_tag, "participant_id 与预期不一致"))
        if row.get("scenario") != args.expected_scenario:
            issues.append(ValidationIssue("warning", sample_tag, "scenario 与预期不一致"))
        if row.get("distance") != args.expected_distance:
            issues.append(ValidationIssue("warning", sample_tag, "distance 与预期不一致"))

        meta_path = meta_dir / f"{sample_tag}.json"
        if not meta_path.exists():
            issues.append(ValidationIssue("error", sample_tag, f"缺少 meta: {meta_path}"))
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        npz_path = resolve_npz_path(row.get("npz_path", ""), exports_ftu, sample_tag)
        if not npz_path.exists():
            issues.append(ValidationIssue("error", sample_tag, f"缺少 npz: {npz_path}"))
            continue

        # 结构层检查：读取 npz 头
        try:
            headers = read_npz_headers(npz_path)
        except Exception as e:  # noqa: BLE001
            issues.append(ValidationIssue("error", sample_tag, f"npz 头解析失败: {e}"))
            continue

        required_keys = {"radar", "time", "heart_rate", "respiration_rate"}
        if set(headers.keys()) != required_keys:
            issues.append(
                ValidationIssue(
                    "error",
                    sample_tag,
                    f"npz keys 不匹配: {sorted(headers.keys())}",
                )
            )
            continue

        radar_shape = tuple(headers["radar"]["shape"])
        time_shape = tuple(headers["time"]["shape"])
        hr_shape = tuple(headers["heart_rate"]["shape"])
        rr_shape = tuple(headers["respiration_rate"]["shape"])

        if len(time_shape) != 1 or len(hr_shape) != 1 or len(rr_shape) != 1:
            issues.append(ValidationIssue("error", sample_tag, "time/heart_rate/respiration_rate 必须为1D"))
        if time_shape != hr_shape or time_shape != rr_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    sample_tag,
                    f"time/hr/rr shape 不一致: {time_shape}/{hr_shape}/{rr_shape}",
                )
            )

        if tuple(meta.get("radar_shape", [])) != radar_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    sample_tag,
                    f"meta.radar_shape 与 npz 不一致: {meta.get('radar_shape')} vs {radar_shape}",
                )
            )

        manifest_shape = parse_manifest_shape(row.get("radar_shape", ""))
        if manifest_shape != radar_shape:
            issues.append(
                ValidationIssue(
                    "error",
                    sample_tag,
                    f"manifest.radar_shape 与 npz 不一致: {manifest_shape} vs {radar_shape}",
                )
            )

        # dtype 检查（从 npy 头描述符）
        if headers["radar"]["descr"] != "<c8":
            issues.append(
                ValidationIssue(
                    "warning",
                    sample_tag,
                    f"radar dtype 描述符预期 <c8，实际 {headers['radar']['descr']}",
                )
            )
        for key in ("time", "heart_rate", "respiration_rate"):
            if headers[key]["descr"] != "<f4":
                issues.append(
                    ValidationIssue(
                        "warning",
                        sample_tag,
                        f"{key} dtype 描述符预期 <f4，实际 {headers[key]['descr']}",
                    )
                )

        # 语义约束
        if meta.get("dataset") != "FTU":
            issues.append(ValidationIssue("error", sample_tag, "meta.dataset 不是 FTU"))
        if meta.get("radar_axes") != ["frames", "doppler", "angle", "range"]:
            issues.append(ValidationIssue("error", sample_tag, "meta.radar_axes 非统一 RDA 语义"))
        if meta.get("radar_preprocess") != "raw_iq_to_rda_fft":
            issues.append(ValidationIssue("warning", sample_tag, "meta.radar_preprocess 非预期值 raw_iq_to_rda_fft"))

        ref_len_meta = int(meta.get("reference_len", -1))
        ref_len_manifest = int(row.get("reference_len", -1))
        ref_len_npz = int(time_shape[0]) if time_shape else -1
        if not (ref_len_meta == ref_len_manifest == ref_len_npz):
            issues.append(
                ValidationIssue(
                    "error",
                    sample_tag,
                    f"reference_len 不一致(meta/manifest/npz): {ref_len_meta}/{ref_len_manifest}/{ref_len_npz}",
                )
            )

    # 数值合理性检查（可选，依赖 numpy）
    np = maybe_import_numpy()
    if np is None:
        issues.append(
            ValidationIssue(
                "warning",
                "GLOBAL",
                "未检测到 numpy，已跳过数值合理性检查（仅完成结构一致性检查）。",
            )
        )
    else:
        for row in rows:
            if row.get("status") != "ok":
                continue
            sample_tag = row["sample_tag"]
            npz_path = resolve_npz_path(row.get("npz_path", ""), exports_ftu, sample_tag)
            if npz_path.exists():
                issues.extend(check_numeric_reasonableness(np, npz_path, sample_tag))

    errors = [x for x in issues if x.level == "error"]
    warnings = [x for x in issues if x.level == "warning"]

    print("=== FTU Export Validation Report ===")
    print(f"exports_ftu: {exports_ftu}")
    print(f"samples_in_manifest: {len(rows)}")
    print(f"errors: {len(errors)}, warnings: {len(warnings)}")

    if warnings:
        print("\n[Warnings]")
        for w in warnings:
            print(f"- [{w.sample_tag}] {w.message}")

    if errors:
        print("\n[Errors]")
        for e in errors:
            print(f"- [{e.sample_tag}] {e.message}")

    report = {
        "exports_ftu": str(exports_ftu),
        "samples_in_manifest": len(rows),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": [x.__dict__ for x in issues],
        "result": "pass" if not errors else "fail",
    }

    if args.json_out is not None:
        try:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\njson_report: {args.json_out}")
        except OSError as e:
            print(f"\n[Warning] JSON 报告写入失败: {args.json_out} ({e})")

    if errors:
        sys.exit(1)
    print("\nResult: PASS")


if __name__ == "__main__":
    main()
