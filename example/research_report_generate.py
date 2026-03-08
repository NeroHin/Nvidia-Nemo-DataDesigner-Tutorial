import data_designer.config as dd
from data_designer.interface import DataDesigner

# =========================================
# 1. 設定模型
# =========================================
# 生成用
MODEL_PROVIDER = "nvidia"
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
MODEL_ALIAS = "nemotron-nano-v3"

# 評估用
JUDGE_MODEL_ALIAS = "qwen-3.5-122b-a10b"
JUDGE_MODEL_ID = "qwen/qwen3.5-122b-a10b" 

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.5,
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

# initialize DataDesigner
dd_app = DataDesigner()
builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# =========================================
# 2. 載入 seed dataset
# =========================================
seed_source = dd.LocalFileSeedSource(path="./data/research_report_seed.json")
builder.with_seed_dataset(seed_source)

# =========================================
# 3. Stage 1 - 建立 outline
#    先把 seed 條件整理成結構化的文章大綱
# =========================================
outline_prompt = """
你是一位企業研究助理，請根據以下條件，生成一份研究型文章的大綱。
請使用繁體中文（台灣用語），並維持分析型、正式、偏技術部落格的風格。

【文章條件】
主題：{{ topic }}
目標讀者：{{ target_audience }}
寫作風格：{{ writing_style }}
文章目標：{{ report_goal }}
語言：{{ language }}
地區：{{ region }}
預期篇幅：{{ expected_length }}
指定章節：{{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
必須包含關鍵字：{{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
限制條件：{{ constraints | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "；") }}

請輸出一份結構化文章大綱，格式如下：
1. title：文章標題
2. thesis：核心論點
3. intro_summary：前言摘要
4. section_plan：依照指定章節逐一列出
   - section_name
   - section_goal
   - key_points（3-5 點）
   - must_include_keywords（本節應帶到哪些關鍵字：{{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}）
   - section_summary_requirement（該節小結要求）
5. conclusion_goal：結語應如何收束
6. target_wording_strategy：如何分配篇幅以符合 {{ expected_length }}

要求：
1. 必須涵蓋 seed 中提供的所有 sections。
2. 每一節都要有明確小結。
3. 不可口語化。
4. 不要捏造真實引用來源。
5. 僅輸出 JSON。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="article_outline",
        model_alias=MODEL_ALIAS,
        prompt=outline_prompt,
    )
)

# =========================================
# 4. Stage 2 - 依章節生成各 section 內容
# =========================================

intro_prompt = """
請根據以下 outline 與文章設定，撰寫文章前言。
使用繁體中文（台灣用語），分析型、正式、偏技術部落格風格。

【文章設定】
主題：{{ topic }}
目標讀者：{{ target_audience }}
寫作風格：{{ writing_style }}
文章目標：{{ report_goal }}
地區：{{ region }}
預期篇幅：{{ expected_length }}
限制條件：{{ constraints | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "；") }}

【文章大綱】
{{ article_outline }}

要求：
1. 作為研究型文章前言，需交代背景、問題意識與本文範圍。
2. 自然帶入主題，但不要提前把全文結論說完。
3. 避免口語化。
4. 不要條列。
5. 約 180-250 字。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="article_intro",
        model_alias=MODEL_ALIAS,
        prompt=intro_prompt,
    )
)

section_body_prompt = """
請根據以下 seed 條件與 outline，撰寫完整文章主體內容。
請依照 seed 中提供的章節順序展開，並確保每一節最後都有一段明確小結。

【文章設定】
主題：{{ topic }}
目標讀者：{{ target_audience }}
寫作風格：{{ writing_style }}
文章目標：{{ report_goal }}
語言：{{ language }}
地區：{{ region }}
預期篇幅：{{ expected_length }}
指定章節：{{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
必須包含關鍵字：{{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
限制條件：{{ constraints | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "；") }}

【文章大綱】
{{ article_outline }}

要求：
1. 必須依序包含以下章節：
   {{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
2. 每一節都要有清楚的小標與小結。
3. 文風要正式、分析型，不可過度口語化。
4. 關鍵字必須自然融入內容，不要硬塞。
5. 內容要像企業內部分享用的研究型文章，而不是行銷文。
6. 不要使用過多條列，盡量以段落撰寫。
7. 不要加入虛構案例、虛構數據或虛構引用。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="article_sections",
        model_alias=MODEL_ALIAS,
        prompt=section_body_prompt,
    )
)

conclusion_prompt = """
請根據以下文章設定與前文內容，撰寫結語。

【文章設定】
主題：{{ topic }}
目標讀者：{{ target_audience }}
寫作風格：{{ writing_style }}
文章目標：{{ report_goal }}
預期篇幅：{{ expected_length }}

【文章大綱】
{{ article_outline }}

【前文內容】
前言：
{{ article_intro }}

主體：
{{ article_sections }}

要求：
1. 收束全文論點。
2. 呼應企業導入情境與實務建議。
3. 不要只是重複前文，要有總結性提升。
4. 使用繁體中文（台灣用語）。
5. 不要條列。
6. 約 120-220 字。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="article_conclusion",
        model_alias=MODEL_ALIAS,
        prompt=conclusion_prompt,
    )
)

# =========================================
# 5. Stage 3 - 匯整與格式化
# =========================================
format_prompt = """
請把以下內容整理成一篇完整、可讀性高的研究型文章 Markdown。

【文章資訊】
主題：{{ topic }}
目標讀者：{{ target_audience }}
寫作風格：{{ writing_style }}
文章目標：{{ report_goal }}
預期篇幅：{{ expected_length }}

【大綱】
{{ article_outline }}

【前言】
{{ article_intro }}

【主體】
{{ article_sections }}

【結語】
{{ article_conclusion }}

格式要求：
1. 補上一個合適的文章標題。
2. 使用 Markdown 標題階層。
3. 保留正式、分析型、偏技術部落格風格。
4. 各 section 都必須有小結。
5. 不要額外虛構引用或參考資料。
6. 只輸出文章本體。
"""

builder.add_column(
    dd.LLMTextColumnConfig(
        name="long_context_case_report",
        model_alias=MODEL_ALIAS,
        prompt=format_prompt,
    )
)

# =========================================
# 6. 品質檢查：字數 / 關鍵字覆蓋 / LLM judge
# =========================================

builder.add_column(
    dd.ExpressionColumnConfig(
        name="report_char_count",
        expr="{{ long_context_case_report | length }}",
        dtype="int",
    )
)

builder.add_column(
    dd.ExpressionColumnConfig(
        name="keyword_check_hint",
        expr="""
主題關鍵字：
{{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}

請人工或後續 validator 檢查以上關鍵字是否都有自然出現。
""".strip(),
    )
)

# LLM judges: outline quality (0-3)

outline_quality_prompt = """
Evaluate the quality of this article outline for the research topic.
Outline:
{{ article_outline }}
Topic: {{ topic }} / {{ report_goal }}
Required sections: {{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
Must include keywords: {{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
please use traditional chinese (taiwan) for the output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="outline_quality_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=outline_quality_prompt,
        scores=[
            dd.Score(
                name="outline_quality",
                description="Overall quality of the article outline: structure completeness, section coverage, key points clarity",
                options={3: "Excellent", 2: "Good", 1: "Fair", 0: "Poor"},
            ),
        ],
    )
)

# LLM judges: section content quality (0-3)
section_content_quality_prompt = """
Evaluate the quality of this article section content for the research topic.
Section:
{{ article_sections }}
Topic: {{ topic }} / {{ report_goal }}
Required sections: {{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
Must include keywords: {{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
please use traditional chinese (taiwan) for the output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="section_content_quality_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=section_content_quality_prompt,
        scores=[
            dd.Score(
                name="section_content_quality",
                description="Overall quality of the article section content: structure completeness, key points clarity",
                options={3: "Excellent", 2: "Good", 1: "Fair", 0: "Poor"},
            ),
        ],
    )
)

# LLM judges: conclusion quality (0-3)
conclusion_quality_prompt = """
Evaluate the quality of this article conclusion for the research topic.
Conclusion:
{{ article_conclusion }}
Topic: {{ topic }} / {{ report_goal }}
Required sections: {{ sections | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
Must include keywords: {{ must_include_keywords | string | replace("'", "") | replace("[", "") | replace("]", "") | replace(", ", "、") }}
please use traditional chinese (taiwan) for the output.
"""

builder.add_column(
    dd.LLMJudgeColumnConfig(
        name="conclusion_quality_score",
        model_alias=JUDGE_MODEL_ALIAS,
        prompt=conclusion_quality_prompt,
        scores=[
            dd.Score(
                name="conclusion_quality",
                description="Overall quality of the article conclusion: structure completeness, key points clarity",

                options={3: "Excellent", 2: "Good", 1: "Fair", 0: "Poor"},
            ),
        ],
    )
)


# =========================================
# 7. 預覽
# =========================================
# dd_preview = dd_app.preview(builder, num_records=1)
# dd_preview.display_sample_record()

# =========================================
# 8. 大量生成
# =========================================
dd_result = dd_app.create(
    builder,
    num_records=1,
    dataset_name="research-article-seed-driven-dataset"
)

# =========================================
# 9. 載入分析報告
# =========================================
dd_analysis = dd_result.load_analysis()
dd_analysis.to_report()