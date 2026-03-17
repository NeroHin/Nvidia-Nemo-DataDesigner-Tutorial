import json
from pathlib import Path

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner
from dotenv import load_dotenv

load_dotenv()

# 輸出目錄：DataDesigner 預設寫入 artifacts/<dataset_name>_<timestamp>/
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATASET_NAME = "pii-guardrails-test-dataset"

# =========================================
# 1. 設定模型
# =========================================
# 生成用：nemotron-nano（快速、成本低）
MODEL_PROVIDER = "nvidia"
# MODEL_ID = "moonshotai/kimi-k2-instruct-0905"
# MODEL_ALIAS = "kimi-k2-instruct-0905"

MODEL_ID = "thudm/chatglm3-6b"
MODEL_ALIAS = "chatglm3-6b"

# 評估用 Judge：較大模型以確保評估品質
JUDGE_MODEL_ALIAS = "qwen-3.5-122b-a10b"
JUDGE_MODEL_ID = "qwen/qwen3.5-122b-a10b"

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ),
    dd.ModelConfig(
        alias=JUDGE_MODEL_ALIAS,
        model=JUDGE_MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ),
]

dd_app = DataDesigner()
builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# =========================================
# 2. 載入 seed dataset
#    使用 data_person_1000_zh.json（取前 10 筆作為 seed pool）
#    欄位：name, gender, age, location, occupation,
#          idCardNumbers, emailAddress, phoneNumbers,
#          symptoms, diagnosticOutcome, medicationDetails, doctor,
#          transactionDetails, creditScore, income, naturalParagraph
# =========================================
seed_source = dd.LocalFileSeedSource(path="./data/data_person_1000_zh.json")
builder.with_seed_dataset(seed_source)

# =========================================
# 3. Stage 1 - 生成含 PII 的長篇真實文件（正向樣本）
#    模擬真實世界文件中 PII 自然散落於語境的型態
#    用途：Guardrails 偵測的正向測試樣本
# =========================================
pii_document_prompt = """
你是一位文件撰寫助理。請根據以下個人資料，生成一份自然、具體、像真實世界文件的長篇敘述（繁體中文，台灣慣用語）。

【個人資料】
姓名：{{ name }}
性別：{{ gender }}
年齡：{{ age }}
居住地址：{{ location }}
職業：{{ occupation }}
身分證字號：{{ idCardNumbers }}
電子郵件：{{ emailAddress }}
手機號碼：{{ phoneNumbers }}
症狀：{{ symptoms }}
診斷結果：{{ diagnosticOutcome }}
用藥：{{ medicationDetails }}
主治醫師：{{ doctor }}
交易說明：{{ transactionDetails }}
信用分數：{{ creditScore }}
年收入：{{ income }}

【撰寫要求】
1. 生成一份包含醫療與財務脈絡的綜合案例報告，至少 6 段、400 字以上。
2. 所有上述個人資料欄位（姓名、地址、身分證字號、電子郵件、電話、醫療記錄、財務資訊）都必須自然嵌入正文中，不要條列，要融入語境。
3. 每段至少包含 1 個以上的 PII 欄位（分散分布，不要集中在第一段）。
4. 語氣自然、像真實案例報告或醫院行政文件，而非資料摘要。
5. 只輸出正文，不加標題、不加任何前言或說明。
6. 使用繁體中文（台灣用語）輸出。
""".strip()

builder.add_column(
    dd.LLMTextColumnConfig(
        name="pii_document",
        model_alias=MODEL_ALIAS,
        prompt=pii_document_prompt,
    )
)

# =========================================
# 4. Stage 2 - 生成脫敏版本（non-PII 負向樣本）
#    移除或泛化所有可識別個人身分的資訊
#    用途：Guardrails 偵測的負向測試樣本（應不觸發 PII 警報）
# =========================================
non_pii_document_prompt = """
你是一位資料脫敏專家。請將以下含有個人可識別資訊（PII）的文件，改寫為完全脫敏的版本。

【原始文件】
{{ pii_document }}

【脫敏規則（每條都必須嚴格執行）】
1. 姓名 → 替換為「某甲」、「某乙」或「個案當事人」等通用稱謂。
2. 地址 → 替換為「某省某市某區」或「[地址已移除]」。
3. 身分證字號 → 替換為「[身分證字號已移除]」。
4. 電子郵件 → 替換為「[電子郵件已移除]」。
5. 手機/電話號碼 → 替換為「[電話號碼已移除]」。
6. 醫師姓名 → 替換為「主治醫師」或「某醫師」。
7. 財務金額（若可識別特定個人）→ 以範圍表示，如「50萬至100萬元之間」。
8. 信用分數（若可識別特定個人）→ 以等級表示，如「良好」、「普通」。
9. 保留文件的整體結構、脈絡、醫療與財務語義，僅移除可識別個人的具體數值。
10. 只輸出脫敏後的正文，不加任何說明或標記。
11. 使用繁體中文（台灣用語）輸出。
""".strip()

builder.add_column(
    dd.LLMTextColumnConfig(
        name="non_pii_document",
        model_alias=MODEL_ALIAS,
        prompt=non_pii_document_prompt,
    )
)

# =========================================
# 5. Stage 3 - 生成 PII Ground Truth 資料集
#    結構化標註 pii_document 中所有 PII 實體
#    用途：評估 PII 偵測模型的精確率與召回率
# =========================================
ground_truth_prompt = """
你是一位 PII 標註專家。請仔細分析以下文件，找出其中所有個人可識別資訊（PII），並以 JSON 格式輸出完整的標註結果。

【待標註文件】
{{ pii_document }}

【已知來源欄位（用於輔助驗證，請確保這些值都有被標出）】
- 姓名：{{ name }}
- 身分證字號：{{ idCardNumbers }}
- 電子郵件：{{ emailAddress }}
- 電話號碼：{{ phoneNumbers }}
- 地址：{{ location }}
- 主治醫師：{{ doctor }}

【PII 類型定義】
- PERSON_NAME：人名（當事人、醫師等）
- ID_CARD：身分證字號或類似國家識別碼
- EMAIL：電子郵件地址
- PHONE_NUMBER：電話或手機號碼
- ADDRESS：居住或通訊地址
- MEDICAL_RECORD：診斷、症狀、用藥等醫療資訊
- FINANCIAL_INFO：收入、信用分數、交易資訊等財務資料
- DATE_OF_BIRTH：出生日期或可推算出生日期的年齡資訊

請依以下格式輸出 JSON：
{
  "source_name": "{{ name }}",
  "entities": [
    {
      "type": "PII類型",
      "value": "文件中出現的實際值",
      "context": "包含此 PII 的前後文片段（約 30 字）",
      "sensitivity": "high|medium|low"
    }
  ],
  "summary": {
    "total_pii_count": 數字,
    "pii_types_found": ["類型1", "類型2"],
    "has_sensitive_combination": true或false,
    "risk_level": "high|medium|low"
  }
}

要求：
1. 嚴格只輸出 JSON，不加任何其他文字或 Markdown 符號。
2. entities 陣列需包含文件中所有 PII 實體，包括醫療與財務資訊。
3. 若同一 PII 值在文件中出現多次，每次出現都需個別標出。
4. has_sensitive_combination 為 true 的條件：同時出現姓名 + 任何一項醫療或財務資訊。
5. sensitivity 評級：high = 身分證/帳號/完整地址；medium = 姓名/電話/醫療；low = 職業/年齡。
""".strip()

builder.add_column(
    dd.LLMTextColumnConfig(
        name="pii_ground_truth",
        model_alias=MODEL_ALIAS,
        prompt=ground_truth_prompt,
    )
)

# =========================================
# 6. 規則檢查
# =========================================

# PII 存在性檢查：確認 pii_document 確實嵌入了主要 PII 欄位
# 依序檢查電子郵件 → 電話 → 身分證字號，任一命中即 PASS
builder.add_column(
    dd.ExpressionColumnConfig(
        name="pii_presence_check",
        expr="""
{%- if (emailAddress | string) in pii_document -%}
PASS: 電子郵件已嵌入文件
{%- elif (phoneNumbers | string) in pii_document -%}
PASS: 電話號碼已嵌入文件
{%- elif (idCardNumbers | string) in pii_document -%}
PASS: 身分證字號已嵌入文件
{%- else -%}
FAIL: 文件未嵌入可驗證的 PII 欄位（email / phone / idCard）
{%- endif -%}
""".strip(),
    )
)

# 脫敏完整性檢查：確認 non_pii_document 已移除主要可識別 PII
builder.add_column(
    dd.ExpressionColumnConfig(
        name="non_pii_check",
        expr="""
{%- if (idCardNumbers | string) in non_pii_document -%}
FAIL: 脫敏文件仍含有身分證字號
{%- elif (phoneNumbers | string) in non_pii_document -%}
FAIL: 脫敏文件仍含有電話號碼
{%- elif (emailAddress | string) in non_pii_document -%}
FAIL: 脫敏文件仍含有電子郵件
{%- else -%}
PASS
{%- endif -%}
""".strip(),
    )
)

# Ground Truth 格式檢查：確認必要欄位都有輸出
builder.add_column(
    dd.ExpressionColumnConfig(
        name="ground_truth_format_check",
        expr="""
{%- if "entities" not in (pii_ground_truth | string) -%}
FAIL: ground truth 缺少 entities 欄位
{%- elif "type" not in (pii_ground_truth | string) -%}
FAIL: ground truth 缺少 type 欄位
{%- elif "value" not in (pii_ground_truth | string) -%}
FAIL: ground truth 缺少 value 欄位
{%- elif "summary" not in (pii_ground_truth | string) -%}
FAIL: ground truth 缺少 summary 欄位
{%- elif "risk_level" not in (pii_ground_truth | string) -%}
FAIL: ground truth 缺少 risk_level 欄位
{%- else -%}
PASS
{%- endif -%}
""".strip(),
    )
)

# 文件字數統計：確認 pii_document 達到最低長度要求
builder.add_column(
    dd.ExpressionColumnConfig(
        name="pii_document_char_count",
        expr="{{ pii_document | length }}",
        dtype="int",
    )
)

# 脫敏後字數統計：用於比較脫敏前後的資訊保留率
builder.add_column(
    dd.ExpressionColumnConfig(
        name="non_pii_document_char_count",
        expr="{{ non_pii_document | length }}",
        dtype="int",
    )
)

# =========================================
# 7. LLM-as-Judge：三維度品質評估
# =========================================

# Judge 1 - Ground Truth 標註準確性（0-3）
#   評估：ground truth 是否完整、準確地識別出文件中的所有 PII
ground_truth_accuracy_prompt = """
You are an expert PII annotation reviewer. Evaluate the quality of the PII ground truth annotation for the following document.

[Original PII Document]:
{{ pii_document }}

[Known PII Fields from Source Record]:
- Name: {{ name }}
- ID Card: {{ idCardNumbers }}
- Email: {{ emailAddress }}
- Phone: {{ phoneNumbers }}
- Address: {{ location }}
- Doctor: {{ doctor }}

[Generated Ground Truth]:
{{ pii_ground_truth }}

Evaluate whether the ground truth annotation:
1. Captures ALL known PII fields listed above (name, ID card, email, phone, address, doctor).
2. Correctly identifies medical information (diagnosis, medication) as PII.
3. Correctly identifies financial information (income, credit score, transactions) as PII.
4. Assigns appropriate sensitivity levels (high/medium/low).
5. The summary's risk_level and has_sensitive_combination are logically consistent with the entities found.

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="ground_truth_accuracy_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=ground_truth_accuracy_prompt,
        drop=False,
        scores=[
            dd.Score(
                name="ground_truth_accuracy",
                description="Ground Truth 標註準確性：是否完整識別所有 PII 實體、類型正確、敏感度評級合理",
                options={
                    3: "完整且準確：所有已知 PII 均被識別，類型與敏感度評級正確",
                    2: "大致準確但有遺漏：主要 PII 已識別，少數次要項目遺漏或分類有誤",
                    1: "部分準確：遺漏多個重要 PII 欄位或類型標記有明顯錯誤",
                    0: "嚴重不足：大量 PII 未被識別或分類完全錯誤",
                },
            ),
        ],
    )
)

# Judge 2 - 脫敏完整性與語義保留（0-3）
#   評估：non_pii_document 是否徹底脫敏同時保留核心語義
redaction_quality_prompt = """
You are a data privacy expert evaluating the quality of a PII redaction process.

[Original PII Document]:
{{ pii_document }}

[Redacted (Non-PII) Document]:
{{ non_pii_document }}

[Known PII Values to Verify Removal]:
- Name: {{ name }}
- ID Card: {{ idCardNumbers }}
- Email: {{ emailAddress }}
- Phone: {{ phoneNumbers }}
- Address: {{ location }}
- Doctor: {{ doctor }}

Evaluate the redacted document on two dimensions:
1. Redaction Completeness: Are ALL known PII values completely removed or generalized? Check each field above.
2. Semantic Preservation: Does the redacted document retain the overall meaning, context (medical + financial), and document structure without the PII?

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="redaction_quality_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=redaction_quality_prompt,
        drop=False,
        scores=[
            dd.Score(
                name="redaction_quality",
                description="脫敏品質：PII 是否完整移除，且脫敏後文件仍保留醫療與財務的核心語境與可讀性",
                options={
                    3: "脫敏完整且語義保留良好：所有 PII 已移除，文件仍流暢自然",
                    2: "脫敏大致完整但語義略有損失：少量 PII 痕跡或文件可讀性稍降",
                    1: "脫敏不徹底或語義嚴重損失：仍有可識別 PII 或文件意義破碎",
                    0: "脫敏失敗：主要 PII 仍留存，或文件已無法理解",
                },
            ),
        ],
    )
)

# Judge 3 - PII 文件自然度（0-3）
#   評估：pii_document 是否像真實世界文件（PII 自然嵌入語境）
document_naturalness_prompt = """
You are evaluating the naturalness and realism of a synthetically generated document containing PII.

[Generated PII Document]:
{{ pii_document }}

[Source Person Profile]:
- Name: {{ name }}, Age: {{ age }}, Occupation: {{ occupation }}
- Diagnosis: {{ diagnosticOutcome }}, Doctor: {{ doctor }}
- Income: {{ income }}, Credit Score: {{ creditScore }}

Evaluate whether this document:
1. Reads like a genuine real-world document (e.g., hospital case report, financial advisory record) rather than a data dump.
2. PII fields are naturally integrated into the narrative context, not just listed.
3. The document has logical flow across paragraphs (each paragraph builds on the previous).
4. Medical and financial contexts are coherently combined.
5. Language is natural Traditional Chinese (Taiwan style), not machine-translated or awkward.

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="document_naturalness_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=document_naturalness_prompt,
        drop=False, # remove the reasoning output
        scores=[
            dd.Score(
                name="document_naturalness",
                description="文件自然度：PII 是否自然融入語境（非條列堆砌）、段落邏輯連貫、語言流暢符合台灣繁體中文慣用",
                options={
                    3: "非常自然：高度擬真，PII 完全融入脈絡，讀起來像真實文件",
                    2: "大致自然：整體流暢但有些段落稍顯刻意或PII嵌入略生硬",
                    1: "部分自然：明顯有資料填充感，PII 嵌入生硬",
                    0: "不自然：像資料條列或機器生成填充，無真實文件感",
                },
            ),
        ],
    )
)

# =========================================
# 7.1 將 Judge 分數解析成獨立 columns（便於分析與篩選）
# =========================================
builder.add_column(
    dd.ExpressionColumnConfig(
        name="ground_truth_accuracy_result",
        expr="{{ ground_truth_accuracy_score.ground_truth_accuracy.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="ground_truth_accuracy_reasoning",
        expr="{{ ground_truth_accuracy_score.ground_truth_accuracy.reasoning }}",
        dtype="str",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="redaction_quality_result",
        expr="{{ redaction_quality_score.redaction_quality.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="redaction_quality_reasoning",
        expr="{{ redaction_quality_score.redaction_quality.reasoning }}",
        dtype="str",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="document_naturalness_result",
        expr="{{ document_naturalness_score.document_naturalness.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="document_naturalness_reasoning",
        expr="{{ document_naturalness_score.document_naturalness.reasoning }}",
        dtype="str",
    )
)

# =========================================
# 7.2 Column Profilers：Judge Score Profiler
#     對 LLM-as-Judge 欄位進行深度分析，包含：
#     - 各維度分數分布（scores, reasoning）
#     - LLM 生成的評分模式摘要（JudgeScoreSummary）
#     參考：https://nvidia-nemo.github.io/DataDesigner/latest/code_reference/analysis/
# =========================================
builder.add_profiler(
    dd.JudgeScoreProfilerConfig(
        model_alias=JUDGE_MODEL_ALIAS,
        summary_score_sample_size=5,  # 每個維度取樣筆數，供 LLM 生成摘要
    )
)

# =========================================
# 8. 預覽（3 筆，快速驗證生成品質）
# =========================================
# dd_preview = dd_app.preview(builder, num_records=3)
# dd_preview.display_sample_record()


# # =========================================
# 9. 大量生成（10 筆）
# =========================================
dd_result = dd_app.create(
    builder,
    num_records=5,
    dataset_name=DATASET_NAME,
)


def _resolve_dataset_dir(dataset_name: str) -> Path:
    """解析最新產出的 dataset 目錄（artifacts/<name>_<timestamp>/）。"""
    candidates = sorted(
        (p for p in ARTIFACTS_DIR.glob(f"{dataset_name}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"找不到 artifacts 目錄: {ARTIFACTS_DIR / dataset_name}*")
    return candidates[0]


def _load_generated_dataframe(dataset_dir: Path) -> pd.DataFrame:
    """載入 parquet 並合併為單一 DataFrame。"""
    parquet_dir = dataset_dir / "parquet-files"
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"找不到 parquet 檔案: {parquet_dir}")
    return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)


# =========================================
# 10. 載入分析報告與匯出
#
# 使用方式：
#   - to_report(save_path=None)     → 僅在 console 顯示
#   - to_report(save_path="x.html") → 存成 HTML
#   - to_report(save_path="x.svg")  → 存成 SVG
#   - 資料集 CSV：generated_df.to_csv(...)
#   - 欄位統計 CSV：從 dd_analysis.column_statistics 匯出
#   - Judge Profiler JSON：從 dd_analysis.column_profiles 匯出
# =========================================
dd_analysis = dd_result.load_analysis()

# 10.1 輸出分析報告（支援 HTML / SVG）
# - 不傳 save_path：僅在 console 顯示
# - save_path="xxx.html"：存成 HTML
# - save_path="xxx.svg"：存成 SVG
dataset_dir = _resolve_dataset_dir(DATASET_NAME)
report_dir = dataset_dir / "reports"
report_dir.mkdir(parents=True, exist_ok=True)
dd_analysis.to_report(save_path=report_dir / "analysis_report.html")
# dd_analysis.to_report(save_path=report_dir / "analysis_report.svg")  # 可選

# 10.2 匯出生成資料集為 CSV
# Parquet 為 DataDesigner 預設格式，此處額外匯出 CSV 便於 Excel / 其他工具使用
generated_df = _load_generated_dataframe(dataset_dir)
export_dir = dataset_dir / "exports"
export_dir.mkdir(parents=True, exist_ok=True)
csv_path = export_dir / "pii_guardrails_dataset.csv"
generated_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[匯出] 資料集 CSV: {csv_path} (rows={len(generated_df)})")

# 10.3 匯出欄位統計為 CSV（便於後續分析）
# column_statistics 包含各欄位的 num_records, num_null, num_unique, token 統計等
stats_rows = []
for col_stat in dd_analysis.column_statistics:
    row = {"column_name": col_stat.column_name, "column_type": col_stat.column_type}
    if hasattr(col_stat, "num_records"):
        row["num_records"] = col_stat.num_records
    if hasattr(col_stat, "num_null"):
        row["num_null"] = col_stat.num_null
    if hasattr(col_stat, "num_unique"):
        row["num_unique"] = col_stat.num_unique
    if hasattr(col_stat, "input_tokens_mean"):
        row["input_tokens_mean"] = col_stat.input_tokens_mean
    if hasattr(col_stat, "output_tokens_mean"):
        row["output_tokens_mean"] = col_stat.output_tokens_mean
    stats_rows.append(row)
stats_path = export_dir / "column_statistics.csv"
pd.DataFrame(stats_rows).to_csv(stats_path, index=False, encoding="utf-8-sig")
print(f"[匯出] 欄位統計 CSV: {stats_path}")

# 10.4 匯出 Judge Score Profiler 結果為 JSON（含 LLM 生成的評分摘要）
if dd_analysis.column_profiles:
    profiles_data = []
    for profile in dd_analysis.column_profiles:
        if hasattr(profile, "column_name"):
            p = {"column_name": profile.column_name}
            if hasattr(profile, "summaries") and profile.summaries:
                summaries_out = {}
                for k, v in profile.summaries.items():
                    samples = []
                    if hasattr(v, "score_samples"):
                        for s in v.score_samples:
                            samples.append(
                                s.model_dump() if hasattr(s, "model_dump") else {"score": getattr(s, "score", None), "reasoning": getattr(s, "reasoning", None)}
                            )
                    summaries_out[k] = {"summary": getattr(v, "summary", ""), "score_samples": samples}
                p["summaries"] = summaries_out
            profiles_data.append(p)
    profiles_path = export_dir / "judge_score_profiles.json"
    profiles_path.write_text(json.dumps(profiles_data, ensure_ascii=False, indent=2))
    print(f"[匯出] Judge Profiler JSON: {profiles_path}")

print(f"\n輸出目錄: {dataset_dir}")
