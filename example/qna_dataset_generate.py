import data_designer.config as dd
from data_designer.interface import DataDesigner

# =========================================
# 1. 設定模型
# =========================================
# 生成用
MODEL_PROVIDER = "nvidia"
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
MODEL_ALIAS = "nemotron-nano-v3"

# 評估用（Judge）
JUDGE_MODEL_ALIAS = "qwen-3.5-122b-a10b"
JUDGE_MODEL_ID = "qwen/qwen3.5-122b-a10b"

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
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
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    ),
]

dd_app = DataDesigner()
builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# =========================================
# 2. 載入 seed dataset
# =========================================
seed_source = dd.LocalFileSeedSource(path="./data/qna_dataset_seed.json")
builder.with_seed_dataset(seed_source)

# =========================================
# 3. Stage 1 - 生成使用者問句
#    根據 persona / intent / tone / edge_case 產出帶有邊界情境的真實問句
# =========================================
user_question_prompt = """
你是資料生成助理，請根據以下條件，模擬一位真實使用者在客服情境下發出的問題或陳述。

【使用者設定】
Persona（用戶類型）：{{ persona }}
Intent（意圖）：{{ intent }}
Tone（語氣風格）：{{ tone }}
Channel（通路）：{{ channel }}
Language（語言）：{{ language }}
Edge Case（邊界情境）：{{ edge_case }}

要求：
1. 語氣與措辭必須完整體現 persona 與 tone 的特色。
   - 若 tone 為「客氣但有點不耐煩」，問句需帶出隱約的急促感。
   - 若 tone 為「直接、帶有抱怨」，問句可帶有負面情緒字眼。
   - 若 tone 為「正式、謹慎」，問句應使用書面語、結構清楚。
2. 邊界情境（{{ edge_case }}）必須自然嵌入問句中，不要點名說明，而是讓它成為問句的一部分。
   - 例如「沒有提供卡別資訊」應表現為：使用者問了問題但沒說是哪張卡。
3. 語言使用 {{ language }}（繁體中文、台灣慣用語）。
4. 只輸出使用者問句本身，不加任何前言、標籤或解釋。
5. 長度控制在 40-120 字。
6. 不使用條列格式。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="user_question",
        model_alias=MODEL_ALIAS,
        prompt=user_question_prompt,
    )
)

# =========================================
# 4. Stage 2 - 生成客服回覆
#    根據使用者問句 + expected_answer_style + policy_constraints 產出合規客服回覆
# =========================================
agent_response_prompt = """
你是一位專業客服助理，請根據以下設定，針對使用者問句給出適當回覆。

【場景設定】
Persona（用戶類型）：{{ persona }}
Difficulty（難度）：{{ difficulty }}
Channel（通路）：{{ channel }}
Expected Answer Style（期望回覆風格）：{{ expected_answer_style }}

【政策限制（每一條都必須嚴格遵守）】
{{ policy_constraints | string | replace("'", "") | replace("[", "- ") | replace("]", "") | replace(", ", "\n- ") }}

【使用者問句】
{{ user_question }}

【邊界情境說明】
{{ edge_case }}

要求：
1. 回覆風格必須符合 expected_answer_style（{{ expected_answer_style }}）。
2. 嚴格遵守所有政策限制，一條都不可違反。
3. 針對邊界情境（{{ edge_case }}），必須給出明確處理：
   - 若使用者缺少必要資訊，主動以友善方式詢問。
   - 若情況涉及敏感資料，引導至正確管道而非直接回答。
4. 回覆語氣需因應 persona（{{ persona }}）調整：
   - 若對方激動或有抱怨，先同理再說明。
   - 若對方謹慎正式，維持同等專業語氣。
5. 語言使用繁體中文（台灣慣用語）。
6. 只輸出客服回覆本身，不加前言或解釋。
7. 長度控制在 80-200 字。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="agent_response",
        model_alias=MODEL_ALIAS,
        prompt=agent_response_prompt,
    )
)

# =========================================
# 5. Stage 3 - 整合為完整 QnA 對話紀錄
#    彙整 metadata + conversation 為結構化 JSON
# =========================================
format_prompt = """
請將以下客服問答整合成一筆完整的結構化對話紀錄，並以 JSON 格式輸出。

【Metadata】
persona: {{ persona }}
intent: {{ intent }}
tone: {{ tone }}
difficulty: {{ difficulty }}
channel: {{ channel }}
edge_case: {{ edge_case }}
expected_answer_style: {{ expected_answer_style }}

【使用者問句】
{{ user_question }}

【客服回覆】
{{ agent_response }}

【政策限制】
{{ policy_constraints | string | replace("'", "") | replace("[", "") | replace("]", "") }}

請依以下格式輸出 JSON：
{
  "metadata": {
    "persona": "...",
    "intent": "...",
    "tone": "...",
    "difficulty": "...",
    "channel": "...",
    "edge_case": "...",
    "expected_answer_style": "..."
  },
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "policy_constraints": ["...", "...", "..."]
}

要求：
1. 嚴格只輸出 JSON，不加任何其他文字或 Markdown 符號。
2. conversation 的 user.content 填入使用者問句，assistant.content 填入客服回覆。
3. policy_constraints 保留為陣列格式，每條獨立一個字串。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="qna_record",
        model_alias=MODEL_ALIAS,
        prompt=format_prompt,
    )
)

# =========================================
# 6. 規則檢查：PII / 禁詞 / 格式 / 字數
# =========================================

# PII 檢查：客服回覆不應主動要求提供身分證字號或完整卡號
builder.add_column(
    dd.ExpressionColumnConfig(
        name="pii_flag",
        expr="""
{%- if "身分證字號" in agent_response and "不需要" not in agent_response and "無需" not in agent_response -%}
FAIL: 客服回覆可能要求提供身分證字號
{%- elif "完整卡號" in agent_response -%}
FAIL: 客服回覆可能要求提供完整卡號
{%- elif "密碼" in agent_response and "請提供" in agent_response -%}
FAIL: 客服回覆要求提供密碼
{%- else -%}
PASS
{%- endif -%}
""".strip(),
    )
)

# 禁詞檢查：客服回覆不應出現不當承諾或指責性詞語
builder.add_column(
    dd.ExpressionColumnConfig(
        name="forbidden_word_flag",
        expr="""
{%- if "保證退款" in agent_response or "一定退款" in agent_response -%}
FAIL: 發現禁詞（保證/一定退款）
{%- elif "絕對沒問題" in agent_response or "一定沒問題" in agent_response -%}
FAIL: 發現禁詞（絕對/一定沒問題）
{%- elif "謊言" in agent_response or "你在騙人" in agent_response or "是你的錯" in agent_response -%}
FAIL: 發現指責性禁詞
{%- elif "保證通過" in agent_response or "確保核准" in agent_response -%}
FAIL: 發現未確認的承諾性禁詞
{%- else -%}
PASS
{%- endif -%}
""".strip(),
    )
)

# 格式檢查：確認 qna_record 包含必要欄位
# 先用 | string 正規化：避免 DataDesigner 將 JSON 輸出自動解析為 dict 後
# Jinja2 的 `in` 變成 dict key 查找而非字串包含檢查，導致 False FAIL
builder.add_column(
    dd.ExpressionColumnConfig(
        name="format_check",
        expr="""
{%- if "conversation" not in (qna_record | string) or "role" not in (qna_record | string) -%}
FAIL: qna_record 缺少 conversation 或 role 欄位
{%- elif "user" not in (qna_record | string) or "assistant" not in (qna_record | string) -%}
FAIL: qna_record 缺少 user 或 assistant role 值
{%- elif "content" not in (qna_record | string) -%}
FAIL: qna_record 缺少 content 欄位
{%- else -%}
PASS
{%- endif -%}
""".strip(),
    )
)

# 回覆字數統計
builder.add_column(
    dd.ExpressionColumnConfig(
        name="response_char_count",
        expr="{{ agent_response | length }}",
        dtype="int",
    )
)

# =========================================
# 7. LLM-as-a-Judge：三維度品質評估
# =========================================

# Judge 1 - 政策合規性（0-3）
policy_compliance_prompt = """
You are evaluating a customer service QnA pair for policy compliance.

[Persona]: {{ persona }}
[Intent]: {{ intent }}
[Policy Constraints]:
{{ policy_constraints | string | replace("'", "") | replace("[", "- ") | replace("]", "") | replace(", ", "\n- ") }}

[User Question]:
{{ user_question }}

[Agent Response]:
{{ agent_response }}

Evaluate whether the agent response strictly follows ALL policy constraints listed above.
Check specifically:
- Does it avoid fabricating any commitments not confirmed by policy?
- Does it avoid requesting prohibited personal information (e.g., full ID number)?
- Does it avoid blaming or accusing the user?
- Does it remind the user to check official announcements if required?

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="policy_compliance_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=policy_compliance_prompt,
        drop=True,
        scores=[
            dd.Score(
                name="policy_compliance",
                description="政策合規性：客服回覆是否完全遵守所有 policy_constraints，未捏造承諾、未洩漏敏感資訊要求、未指責客戶",
                options={3: "完全合規", 2: "大致合規但有輕微疏漏", 1: "部分違規", 0: "嚴重違規"},
            ),
        ],
    )
)

# Judge 2 - 語氣適切性（0-3）
tone_appropriateness_prompt = """
You are evaluating a customer service QnA pair for tone and style appropriateness.

[Persona]: {{ persona }}
[Tone]: {{ tone }}
[Expected Answer Style]: {{ expected_answer_style }}
[Difficulty]: {{ difficulty }}
[Edge Case]: {{ edge_case }}

[User Question]:
{{ user_question }}

[Agent Response]:
{{ agent_response }}

Evaluate whether the agent response's tone and style is appropriate given:
1. The persona type and their emotional state implied by the tone.
2. The expected answer style specified (e.g., concise & friendly, empathetic & stabilizing, professional & compliance-oriented).
3. The difficulty level and edge case complexity.

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="tone_appropriateness_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=tone_appropriateness_prompt,
        drop=True,
        scores=[
            dd.Score(
                name="tone_appropriateness",
                description="語氣適切性：客服回覆的語氣與風格是否符合 persona 特性、expected_answer_style 以及難度等級的情境需求",
                options={3: "完全適切", 2: "大致適切但有些許偏差", 1: "語氣明顯不符", 0: "完全不適切"},
            ),
        ],
    )
)

# Judge 3 - 回覆有效性（0-3）
response_helpfulness_prompt = """
You are evaluating a customer service QnA pair for response helpfulness and effectiveness.

[Persona]: {{ persona }}
[Intent]: {{ intent }}
[Edge Case]: {{ edge_case }}
[Expected Answer Style]: {{ expected_answer_style }}

[User Question]:
{{ user_question }}

[Agent Response]:
{{ agent_response }}

Evaluate whether the agent response is genuinely helpful:
1. Does it address the user's core intent?
2. Does it handle the edge case gracefully (e.g., ask for missing info, redirect to proper channel)?
3. Does it provide a clear next step or action path for the user?
4. Is it free of vague or non-actionable language?

Please use traditional Chinese (Taiwan) for any reasoning output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="response_helpfulness_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=response_helpfulness_prompt,
        drop=True,
        scores=[
            dd.Score(
                name="response_helpfulness",
                description="回覆有效性：客服回覆是否真正解決使用者意圖、妥善處理 edge case，並提供清晰可執行的下一步行動路徑",
                options={3: "非常有效", 2: "有效但部分缺失", 1: "效果有限", 0: "完全無效"},
            ),
        ],
    )
)

# =========================================
# 7.1 將 Judge 分數解析成獨立 columns（便於分析與篩選）
# =========================================
builder.add_column(
    dd.ExpressionColumnConfig(
        name="policy_compliance_result",
        expr="{{ policy_compliance_score.policy_compliance.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="policy_compliance_reasoning",
        expr="{{ policy_compliance_score.policy_compliance.reasoning }}",
        dtype="str",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="tone_appropriateness_result",
        expr="{{ tone_appropriateness_score.tone_appropriateness.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="tone_appropriateness_reasoning",
        expr="{{ tone_appropriateness_score.tone_appropriateness.reasoning }}",
        dtype="str",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="response_helpfulness_result",
        expr="{{ response_helpfulness_score.response_helpfulness.score }}",
        dtype="int",
    )
)
builder.add_column(
    dd.ExpressionColumnConfig(
        name="response_helpfulness_reasoning",
        expr="{{ response_helpfulness_score.response_helpfulness.reasoning }}",
        dtype="str",
    )
)

# =========================================
# 8. 預覽（可選，取消註解使用）
# =========================================
# dd_preview = dd_app.preview(builder, num_records=1)
# dd_preview.display_sample_record()

# =========================================
# 9. 大量生成
# =========================================
dd_result = dd_app.create(
    builder,
    num_records=3,
    dataset_name="qna-customer-service-dataset"
)

# =========================================
# 10. 載入分析報告
# =========================================
dd_analysis = dd_result.load_analysis()
dd_analysis.to_report()
