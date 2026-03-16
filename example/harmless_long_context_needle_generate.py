from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MODEL_PROVIDER = "nvidia"
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
MODEL_ALIAS = "nemotron-nano-v3"
DATASET_NAME = "harmless-long-context-needle-dataset"
DEFAULT_RPM_LIMIT = 40
DEFAULT_MAX_PARALLEL_REQUESTS = 1
PERSON_SAMPLE_SEED = 42

TARGET_WINDOWS = {
    "1K_context": (900, 1150),
    "4k_context": (3700, 4300),
    "8k_context": (7500, 8500),
    "16k_context": (15000, 17000),
    "32k_context": (31000, 33000),
}

FINAL_COLUMN_ORDER = [
    "1K_context",
    "4k_context",
    "8k_context",
    "16k_context",
    "32k_context",
    "16k_needle_start_context",
    "16k_needle_mid_context",
    "16k_needle_end_context",
    "32k_needle_start_context",
    "32k_needle_mid_context",
    "32k_needle_end_context",
    "needle_type",
    "needle_value",
    "topic_seed",
]

TOPIC_SEEDS = [
    "城市公共圖書館如何透過空間與服務設計提升社區學習參與",
    "大眾運輸資訊系統在尖峰時段的資訊傳遞與乘客引導策略",
    "校園節能治理如何結合設備更新、使用者行為與行政流程",
    "地方文化館舍在常設展與教育活動之間的內容規劃原則",
    "社區回收制度從宣導到執行的流程設計與協作分工",
    "公共公園維護管理如何兼顧生態保育、休閒需求與季節變化",
    "中小型展覽活動的動線規劃、現場營運與觀眾體驗優化方法",
    "地方觀光資訊整合平台如何改善旅遊前、中、後期的資訊落差",
    "社區型終身學習課程在招生、課程安排與成果追蹤上的設計重點",
    "防災宣導內容如何針對不同生活場景建立可執行的應變認知",
]

SEGMENT_SPECS = [
    ("segment_01", 1, 1000),
    ("segment_02", 2, 1500),
    ("segment_03", 3, 1500),
    ("segment_04", 4, 2000),
    ("segment_05", 5, 2000),
    ("segment_06", 6, 2000),
    ("segment_07", 7, 2000),
    ("segment_08", 8, 2000),
    ("segment_09", 9, 2000),
    ("segment_10", 10, 2000),
    ("segment_11", 11, 2000),
    ("segment_12", 12, 2000),
    ("segment_13", 13, 2000),
    ("segment_14", 14, 2000),
    ("segment_15", 15, 2000),
    ("segment_16", 16, 2000),
    ("segment_17", 17, 2000),
]

LLM_REQUESTS_PER_RECORD = 1 + len(SEGMENT_SPECS)
DEFAULT_NUM_RECORDS = max(1, DEFAULT_RPM_LIMIT // LLM_REQUESTS_PER_RECORD)

TAIWAN_ID_RE = re.compile(r"\b[A-Z][12]\d{8}\b")
CN_ID_RE = re.compile(r"\b\d{17}[\dXx]\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?886[- ]?)?09\d{8}\b|\b1\d{10}\b")


@dataclass(frozen=True)
class ValidationResult:
    row_index: int
    passed: bool
    checks: dict[str, object]


@dataclass(frozen=True)
class BatchRunResult:
    batch_index: int
    dataset_name: str
    dataset_dir: Path
    num_records: int
    start_offset: int
    elapsed_seconds: float
    requests_estimate: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return repo_root() / "data"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def runtime_seed_dir() -> Path:
    return artifacts_dir() / "runtime-seeds"


def build_runtime_seed_records(num_records: int, start_offset: int = 0) -> list[dict[str, str]]:
    people_path = data_dir() / "data_person_1000_zh.json"
    people = json.loads(people_path.read_text(encoding="utf-8"))
    rng = random.Random(PERSON_SAMPLE_SEED)
    shuffled_people = list(people)
    rng.shuffle(shuffled_people)

    records: list[dict[str, str]] = []
    for index in range(num_records):
        absolute_index = start_offset + index
        topic = TOPIC_SEEDS[absolute_index % len(TOPIC_SEEDS)]
        person = shuffled_people[absolute_index % len(shuffled_people)]
        needle_type = "name" if absolute_index % 2 == 0 else "id_card"
        needle_value = str(person["name"] if needle_type == "name" else person["idCardNumbers"])
        records.append(
            {
                "record_id": str(absolute_index + 1),
                "topic_seed": topic,
                "needle_type": needle_type,
                "needle_value": needle_value,
            }
        )
    return records


def write_runtime_seed_file(records: list[dict[str, str]], dataset_name: str) -> Path:
    runtime_seed_dir().mkdir(parents=True, exist_ok=True)
    seed_path = runtime_seed_dir() / f"{dataset_name}_seed.json"
    seed_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return seed_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate harmless long contexts with deterministic PII needle insertion.")
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"Dataset name prefix written under artifacts/. Default: {DATASET_NAME}",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=DEFAULT_NUM_RECORDS,
        help=(
            "Number of records to generate in this run. "
            f"Default is auto-clamped for RPM{DEFAULT_RPM_LIMIT}: {DEFAULT_NUM_RECORDS}"
        ),
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=DEFAULT_RPM_LIMIT,
        help=f"LLM API requests-per-minute limit. Default: {DEFAULT_RPM_LIMIT}",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Offset into the deterministic seed stream. Use this to avoid duplicate rows across multiple runs.",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=DEFAULT_MAX_PARALLEL_REQUESTS,
        help=f"Per-column parallel requests for DataDesigner. Default: {DEFAULT_MAX_PARALLEL_REQUESTS}",
    )
    parser.add_argument(
        "--auto-batch-total-records",
        type=int,
        default=0,
        help="Enable auto-batch mode and keep running until this total record count is reached.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Extra fixed sleep between batches in auto-batch mode. Default: 0",
    )
    return parser.parse_args()


def recommended_num_records(rpm_limit: int) -> int:
    return max(1, rpm_limit // LLM_REQUESTS_PER_RECORD)


def build_outline_prompt() -> str:
    return """
你是一位中性說明文規劃編輯。請根據以下主題，規劃一份可延展為超長內容的寫作藍圖。

【主題】
{{ topic_seed }}

【輸出要求】
1. 請規劃 17 個依序展開的小節，節次固定使用 01 到 17。
2. 每一節都要包含：
   - 節次
   - 本節主旨
   - 三個應展開重點
   - 與前後節的銜接提示
3. 全文設定必須維持中性、分析型、資訊性高的說明文風格。
4. 嚴禁出現任何人名、地址、電話、電子郵件、身分證字號、病例、交易資料、帳號、公司內部資訊。
5. 不要寫成故事，不要設定主角，不要虛構訪談或真實人物案例。
6. 使用繁體中文（台灣用語）。
7. 只輸出純文字大綱，不要 JSON、Markdown、表格或任何額外前言。
""".strip()


def build_segment_prompt(section_number: int, target_chars: int) -> str:
    return f"""
你是一位中性說明文撰稿人。請根據以下主題與大綱，只撰寫對應第 {section_number:02d} 節的正文內容。

【主題】
{{{{ topic_seed }}}}

【寫作藍圖】
{{{{ master_outline }}}}

【撰寫要求】
1. 僅展開第 {section_number:02d} 節，不要加入標題、節次標記或條列。
2. 內容要像一篇超長說明文中的連續正文片段，語氣平實、邏輯清楚、資訊密度高。
3. 可自然承接前文，但不要回顧前文內容，也不要預告後文。
4. 不要使用真實人物、真實個案、真實地址、電話、Email、身分證字號、金融帳務或醫療紀錄。
5. 盡量避免出現任何可被解讀為個人可識別資訊的格式。
6. 使用繁體中文（台灣用語）。
7. 長度目標約 {target_chars} 字，允許上下 10% 浮動。
8. 只輸出正文，不加前言或說明。
""".strip()


def concat_expression(column_names: Iterable[str]) -> str:
    names = list(column_names)
    if not names:
        raise ValueError("column_names cannot be empty")
    return "{{ " + " ~ '\\n\\n' ~ ".join(names) + " }}"


def insert_needle(text: str, needle: str, position: str) -> str:
    if position == "start":
        return f"{needle}\n{text}"
    if position == "mid":
        index = len(text) // 2
        return f"{text[:index]}\n{needle}\n{text[index:]}"
    if position == "end":
        return f"{text}\n{needle}"
    raise ValueError(f"Unsupported position: {position}")


def resolve_dataset_dir(dataset_name: str) -> Path:
    candidates = sorted(
        (path for path in artifacts_dir().glob(f"{dataset_name}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Cannot find artifacts for dataset '{dataset_name}' in {artifacts_dir()}")
    return candidates[0]


def load_generated_dataframe(dataset_dir: Path) -> pd.DataFrame:
    parquet_dir = dataset_dir / "parquet-files"
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    frames = [pd.read_parquet(path) for path in parquet_files]
    return pd.concat(frames, ignore_index=True)


def rename_and_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "context_1k": "1K_context",
            "context_4k": "4k_context",
            "context_8k": "8k_context",
            "context_16k": "16k_context",
            "context_32k": "32k_context",
        }
    )

    renamed["16k_needle_start_context"] = renamed.apply(
        lambda row: insert_needle(row["16k_context"], row["needle_value"], "start"), axis=1
    )
    renamed["16k_needle_mid_context"] = renamed.apply(
        lambda row: insert_needle(row["16k_context"], row["needle_value"], "mid"), axis=1
    )
    renamed["16k_needle_end_context"] = renamed.apply(
        lambda row: insert_needle(row["16k_context"], row["needle_value"], "end"), axis=1
    )
    renamed["32k_needle_start_context"] = renamed.apply(
        lambda row: insert_needle(row["32k_context"], row["needle_value"], "start"), axis=1
    )
    renamed["32k_needle_mid_context"] = renamed.apply(
        lambda row: insert_needle(row["32k_context"], row["needle_value"], "mid"), axis=1
    )
    renamed["32k_needle_end_context"] = renamed.apply(
        lambda row: insert_needle(row["32k_context"], row["needle_value"], "end"), axis=1
    )

    return renamed[FINAL_COLUMN_ORDER].copy()


def id_like_pattern_found(text: str) -> bool:
    return bool(TAIWAN_ID_RE.search(text) or CN_ID_RE.search(text))


def harmless_pattern_checks(text: str) -> dict[str, bool]:
    return {
        "contains_id_pattern": id_like_pattern_found(text),
        "contains_email_pattern": bool(EMAIL_RE.search(text)),
        "contains_phone_pattern": bool(PHONE_RE.search(text)),
    }


def count_occurrences(text: str, needle: str) -> int:
    return text.count(needle)


def expected_mid_variant(base_text: str, needle: str) -> str:
    index = len(base_text) // 2
    return f"{base_text[:index]}\n{needle}\n{base_text[index:]}"


def validate_row(row: pd.Series, row_index: int) -> ValidationResult:
    checks: dict[str, object] = {}
    needle = row["needle_value"]

    for column_name, window in TARGET_WINDOWS.items():
        checks[f"{column_name}_length_ok"] = window[0] <= len(row[column_name]) <= window[1]

    for column_name in TARGET_WINDOWS:
        pattern_checks = harmless_pattern_checks(row[column_name])
        checks[f"{column_name}_needle_absent"] = needle not in row[column_name]
        checks[f"{column_name}_id_absent"] = not pattern_checks["contains_id_pattern"]
        checks[f"{column_name}_email_absent"] = not pattern_checks["contains_email_pattern"]
        checks[f"{column_name}_phone_absent"] = not pattern_checks["contains_phone_pattern"]

    base_pairs = [
        ("16k_context", "16k_needle_start_context", "start"),
        ("16k_context", "16k_needle_mid_context", "mid"),
        ("16k_context", "16k_needle_end_context", "end"),
        ("32k_context", "32k_needle_start_context", "start"),
        ("32k_context", "32k_needle_mid_context", "mid"),
        ("32k_context", "32k_needle_end_context", "end"),
    ]

    for base_column, variant_column, position in base_pairs:
        variant_text = row[variant_column]
        base_text = row[base_column]
        checks[f"{variant_column}_single_needle"] = count_occurrences(variant_text, needle) == 1
        if position == "start":
            expected = f"{needle}\n{base_text}"
            checks[f"{variant_column}_exact_match"] = variant_text == expected
            checks[f"{variant_column}_position_ratio"] = 0.0
        elif position == "mid":
            expected = expected_mid_variant(base_text, needle)
            checks[f"{variant_column}_exact_match"] = variant_text == expected
            checks[f"{variant_column}_position_ratio"] = 0.5
        else:
            expected = f"{base_text}\n{needle}"
            checks[f"{variant_column}_exact_match"] = variant_text == expected
            checks[f"{variant_column}_position_ratio"] = 1.0

    passed = all(value for key, value in checks.items() if isinstance(value, bool))
    return ValidationResult(row_index=row_index, passed=passed, checks=checks)


def build_validation_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    results = [validate_row(row, index) for index, row in df.iterrows()]
    detail_rows = [{"row_index": result.row_index, "passed": result.passed, **result.checks} for result in results]
    details_df = pd.DataFrame(detail_rows)

    summary = {
        "dataset_name": DATASET_NAME,
        "row_count": int(len(df)),
        "all_rows_passed": bool(details_df["passed"].all()),
        "passed_rows": int(details_df["passed"].sum()),
        "failed_rows": int((~details_df["passed"]).sum()),
        "length_summary": {
            column_name: {
                "min": int(df[column_name].map(len).min()),
                "max": int(df[column_name].map(len).max()),
                "mean": float(round(df[column_name].map(len).mean(), 2)),
            }
            for column_name in TARGET_WINDOWS
        },
    }
    return details_df, summary


def build_dataset_summary(
    df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_name: str,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "all_rows_passed": bool(validation_df["passed"].all()),
        "passed_rows": int(validation_df["passed"].sum()),
        "failed_rows": int((~validation_df["passed"]).sum()),
        "length_summary": {
            column_name: {
                "min": int(df[column_name].map(len).min()),
                "max": int(df[column_name].map(len).max()),
                "mean": float(round(df[column_name].map(len).mean(), 2)),
            }
            for column_name in TARGET_WINDOWS
        },
    }


def save_final_outputs(dataset_dir: Path, final_df: pd.DataFrame, validation_df: pd.DataFrame, summary: dict[str, object]) -> None:
    output_dir = dataset_dir / "final-dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_parquet(output_dir / "harmless_long_context_needle.parquet", index=False)
    final_df.to_csv(output_dir / "harmless_long_context_needle.csv", index=False)
    validation_df.to_csv(output_dir / "validation_details.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_generation(seed_path: Path, dataset_name: str, num_records: int, max_parallel_requests: int) -> None:
    from dotenv import load_dotenv

    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "找不到 data_designer。請先安裝 `data-designer` 與 `python-dotenv` 後再執行此腳本。"
        ) from exc

    load_dotenv()

    model_configs = [
        dd.ModelConfig(
            alias=MODEL_ALIAS,
            model=MODEL_ID,
            provider=MODEL_PROVIDER,
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_parallel_requests=max_parallel_requests,
                temperature=0.8,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        )
    ]

    dd_app = DataDesigner()
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="master_outline",
            model_alias=MODEL_ALIAS,
            prompt=build_outline_prompt(),
        )
    )

    for segment_name, section_number, target_chars in SEGMENT_SPECS:
        builder.add_column(
            dd.LLMTextColumnConfig(
                name=segment_name,
                model_alias=MODEL_ALIAS,
                prompt=build_segment_prompt(section_number=section_number, target_chars=target_chars),
            )
        )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="context_1k",
            expr=concat_expression(["segment_01"]),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="context_4k",
            expr=concat_expression(["segment_01", "segment_02", "segment_03"]),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="context_8k",
            expr=concat_expression(["segment_01", "segment_02", "segment_03", "segment_04", "segment_05"]),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="context_16k",
            expr=concat_expression(
                [
                    "segment_01",
                    "segment_02",
                    "segment_03",
                    "segment_04",
                    "segment_05",
                    "segment_06",
                    "segment_07",
                    "segment_08",
                    "segment_09",
                ]
            ),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="context_32k",
            expr=concat_expression([segment_name for segment_name, _, _ in SEGMENT_SPECS]),
        )
    )

    for internal_name in ["context_1k", "context_4k", "context_8k", "context_16k", "context_32k"]:
        builder.add_column(
            dd.ExpressionColumnConfig(
                name=f"{internal_name}_char_count",
                expr=f"{{{{ {internal_name} | length }}}}",
                dtype="int",
            )
        )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="base_context_needle_absence_check",
            expr="""
{%- if (needle_value | string) in context_1k -%}
FAIL: context_1k 含有 needle_value
{%- elif (needle_value | string) in context_4k -%}
FAIL: context_4k 含有 needle_value
{%- elif (needle_value | string) in context_8k -%}
FAIL: context_8k 含有 needle_value
{%- elif (needle_value | string) in context_16k -%}
FAIL: context_16k 含有 needle_value
{%- elif (needle_value | string) in context_32k -%}
FAIL: context_32k 含有 needle_value
{%- else -%}
PASS
{%- endif -%}
""".strip(),
            )
        )

    builder.add_column(
        dd.ExpressionColumnConfig(
            name="length_window_check",
            expr="""
{%- if context_1k_char_count < 900 or context_1k_char_count > 1150 -%}
FAIL: context_1k 長度不在目標區間
{%- elif context_4k_char_count < 3700 or context_4k_char_count > 4300 -%}
FAIL: context_4k 長度不在目標區間
{%- elif context_8k_char_count < 7500 or context_8k_char_count > 8500 -%}
FAIL: context_8k 長度不在目標區間
{%- elif context_16k_char_count < 15000 or context_16k_char_count > 17000 -%}
FAIL: context_16k 長度不在目標區間
{%- elif context_32k_char_count < 31000 or context_32k_char_count > 33000 -%}
FAIL: context_32k 長度不在目標區間
{%- else -%}
PASS
{%- endif -%}
""".strip(),
            )
        )

    # dd_preview = dd_app.preview(builder, num_records=2)
    # dd_preview.display_sample_record()

    dd_result = dd_app.create(
        builder,
        num_records=num_records,
        dataset_name=dataset_name,
    )

    try:
        dd_analysis = dd_result.load_analysis()
        dd_analysis.to_report()
    except Exception:
        pass


def process_dataset_outputs(
    dataset_name: str,
    rpm_limit: int,
    max_parallel_requests: int,
    num_records: int,
    start_offset: int,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dataset_dir = resolve_dataset_dir(dataset_name)
    generated_df = load_generated_dataframe(dataset_dir)
    final_df = rename_and_project_columns(generated_df)
    validation_df, validation_summary = build_validation_outputs(final_df)
    validation_summary = build_dataset_summary(final_df, validation_df, dataset_name)
    validation_summary["generation_config"] = {
        "dataset_name": dataset_name,
        "num_records": num_records,
        "rpm_limit": rpm_limit,
        "recommended_num_records": recommended_num_records(rpm_limit),
        "max_parallel_requests": max_parallel_requests,
        "start_offset": start_offset,
        "llm_requests_per_record": LLM_REQUESTS_PER_RECORD,
    }
    save_final_outputs(dataset_dir, final_df, validation_df, validation_summary)
    return dataset_dir, final_df, validation_df, validation_summary


def print_rate_limit_warnings(args: argparse.Namespace, safe_num_records: int) -> None:
    if args.num_records > safe_num_records:
        print(
            "警告：目前設定的單次生成筆數高於 RPM 安全建議值。"
            f" rpm_limit={args.rpm_limit}、每筆約 {LLM_REQUESTS_PER_RECORD} 次 LLM 請求、"
            f"建議 --num-records <= {safe_num_records}，目前為 {args.num_records}。"
        )
    if args.max_parallel_requests > 1:
        print(
            "警告：max_parallel_requests > 1 會提高瞬間請求密度。"
            f" 目前值為 {args.max_parallel_requests}，RPM 受限時建議維持 1。"
        )


def run_single_batch(
    dataset_name: str,
    num_records: int,
    rpm_limit: int,
    start_offset: int,
    max_parallel_requests: int,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object], BatchRunResult]:
    records = build_runtime_seed_records(num_records, start_offset=start_offset)
    seed_path = write_runtime_seed_file(records, dataset_name=dataset_name)
    started_at = time.monotonic()
    run_generation(
        seed_path=seed_path,
        dataset_name=dataset_name,
        num_records=num_records,
        max_parallel_requests=max_parallel_requests,
    )
    elapsed_seconds = time.monotonic() - started_at
    dataset_dir, final_df, validation_df, validation_summary = process_dataset_outputs(
        dataset_name=dataset_name,
        rpm_limit=rpm_limit,
        max_parallel_requests=max_parallel_requests,
        num_records=num_records,
        start_offset=start_offset,
    )
    batch_result = BatchRunResult(
        batch_index=0,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        num_records=num_records,
        start_offset=start_offset,
        elapsed_seconds=elapsed_seconds,
        requests_estimate=num_records * LLM_REQUESTS_PER_RECORD,
    )
    return dataset_dir, final_df, validation_df, validation_summary, batch_result


def ensure_rate_limit_pause(batch_result: BatchRunResult, rpm_limit: int, cooldown_seconds: float) -> None:
    target_seconds = (batch_result.requests_estimate / rpm_limit) * 60 if rpm_limit > 0 else 0
    sleep_seconds = max(0.0, target_seconds - batch_result.elapsed_seconds) + max(0.0, cooldown_seconds)
    if sleep_seconds > 0:
        print(
            f"批次 {batch_result.batch_index:03d} 進入冷卻 {sleep_seconds:.1f} 秒，"
            f"以符合 RPM{rpm_limit}。"
        )
        time.sleep(sleep_seconds)


def save_combined_outputs(
    base_dataset_name: str,
    combined_df: pd.DataFrame,
    combined_validation_df: pd.DataFrame,
    combined_summary: dict[str, object],
    batch_results: list[BatchRunResult],
) -> Path:
    output_dir = artifacts_dir() / f"{base_dataset_name}-combined" / "final-dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_dir / "harmless_long_context_needle.parquet", index=False)
    combined_df.to_csv(output_dir / "harmless_long_context_needle.csv", index=False)
    combined_validation_df.to_csv(output_dir / "validation_details.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(
            [
                {
                    "batch_index": result.batch_index,
                    "dataset_name": result.dataset_name,
                    "dataset_dir": str(result.dataset_dir),
                    "num_records": result.num_records,
                    "start_offset": result.start_offset,
                    "elapsed_seconds": round(result.elapsed_seconds, 2),
                    "requests_estimate": result.requests_estimate,
                }
                for result in batch_results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def run_auto_batch_mode(args: argparse.Namespace) -> None:
    batch_size = args.num_records
    total_records = args.auto_batch_total_records
    batch_results: list[BatchRunResult] = []
    final_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []

    generated_records = 0
    batch_index = 1
    while generated_records < total_records:
        remaining_records = total_records - generated_records
        current_batch_size = min(batch_size, remaining_records)
        batch_dataset_name = f"{args.dataset_name}-batch-{batch_index:03d}"
        current_start_offset = args.start_offset + generated_records
        print(
            f"開始批次 {batch_index:03d}: dataset={batch_dataset_name}, "
            f"records={current_batch_size}, start_offset={current_start_offset}"
        )

        dataset_dir, final_df, validation_df, _, batch_result = run_single_batch(
            dataset_name=batch_dataset_name,
            num_records=current_batch_size,
            rpm_limit=args.rpm_limit,
            start_offset=current_start_offset,
            max_parallel_requests=args.max_parallel_requests,
        )
        batch_results.append(
            BatchRunResult(
                batch_index=batch_index,
                dataset_name=batch_dataset_name,
                dataset_dir=dataset_dir,
                num_records=current_batch_size,
                start_offset=current_start_offset,
                elapsed_seconds=batch_result.elapsed_seconds,
                requests_estimate=batch_result.requests_estimate,
            )
        )
        final_frames.append(final_df)

        validation_copy = validation_df.copy()
        validation_copy["batch_index"] = batch_index
        validation_copy["dataset_name"] = batch_dataset_name
        validation_copy["global_row_index"] = range(generated_records, generated_records + len(validation_copy))
        validation_frames.append(validation_copy)

        generated_records += current_batch_size
        if generated_records < total_records:
            ensure_rate_limit_pause(batch_results[-1], args.rpm_limit, args.cooldown_seconds)
        batch_index += 1

    combined_df = pd.concat(final_frames, ignore_index=True)
    combined_validation_df = pd.concat(validation_frames, ignore_index=True)
    combined_summary = build_dataset_summary(combined_df, combined_validation_df, f"{args.dataset_name}-combined")
    combined_summary["generation_config"] = {
        "mode": "auto_batch",
        "base_dataset_name": args.dataset_name,
        "total_records": total_records,
        "batch_size": batch_size,
        "rpm_limit": args.rpm_limit,
        "recommended_num_records": recommended_num_records(args.rpm_limit),
        "max_parallel_requests": args.max_parallel_requests,
        "initial_start_offset": args.start_offset,
        "cooldown_seconds": args.cooldown_seconds,
        "llm_requests_per_record": LLM_REQUESTS_PER_RECORD,
        "batch_count": len(batch_results),
    }
    output_dir = save_combined_outputs(
        base_dataset_name=args.dataset_name,
        combined_df=combined_df,
        combined_validation_df=combined_validation_df,
        combined_summary=combined_summary,
        batch_results=batch_results,
    )
    print(f"Combined dataset saved to: {output_dir}")
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    safe_num_records = recommended_num_records(args.rpm_limit)
    print_rate_limit_warnings(args, safe_num_records)

    if args.auto_batch_total_records > 0:
        run_auto_batch_mode(args)
        return

    dataset_dir, _, _, validation_summary, _ = run_single_batch(
        dataset_name=args.dataset_name,
        num_records=args.num_records,
        rpm_limit=args.rpm_limit,
        start_offset=args.start_offset,
        max_parallel_requests=args.max_parallel_requests,
    )

    print(f"Final dataset saved to: {dataset_dir / 'final-dataset'}")
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
