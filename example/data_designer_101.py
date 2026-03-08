"""
1. 設定用於合成資料生成（SDG）的模型 
2. 設定用來多元化資料集的種子資料集與欄位 
3. 用提示詞與結構化輸出來設定 LLM 的欄位 
4. 預覽資料集，並根據需要調整設定 
5. 大量產生資料 
6. 評估您的資料品質
"""

import data_designer.config as dd
from data_designer.interface import DataDesigner
from dotenv import load_dotenv      
import time

load_dotenv()

# 1. 設定用於合成資料生成（SDG）的模型 
# This name is set in the model provider configuration.
MODEL_PROVIDER = "nvidia"

# The model ID is from build.nvidia.com.
MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"

# We choose this alias to be descriptive for our use case.
MODEL_ALIAS = "nemotron-nano-v3"

model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
]

dd_app = DataDesigner()
builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

# 2. 匯入外部 JSON 當 seed dataset
seed_source = dd.LocalFileSeedSource(path="./data/data_person_1000_zh.json")
builder.with_seed_dataset(seed_source)

# 3. 用提示詞與結構化輸出來設定 LLM 的欄位
generate_prompt = """
                請根據以下個人資料，撰寫一篇案例敘述（繁體中文，台灣用語）：
                姓名：{{ name }}
                性別：{{ gender }}
                年齡：{{ age }}
                居住地：{{ location }}
                職業：{{ occupation }}
                症狀：{{ symptoms }}
                診斷結果：{{ diagnosticOutcome }}
                用藥：{{ medicationDetails }}
                主治醫師：{{ doctor }}
                交易資訊：{{ transactionDetails }}
                信用分數：{{ creditScore }}
                年收入：{{ income }}

                要求：
                2. 需同時包含醫療與金融脈絡
                3. 內容自然、連貫、像真實案例描述
                4. 不要條列，只輸出正文
                5. 使用正體中文回傳
"""

# 長文本生成欄位
builder.add_column(
    dd.LLMTextColumnConfig(
        name="long_context_case_report",
        model_alias=MODEL_ALIAS,
        prompt=generate_prompt
    )
)

# 4. 預覽資料集，並根據需要調整設定 
dd_preview = dd_app.preview(builder, num_records=2)

# 檢視單筆記錄  
dd_preview.display_sample_record()

# 5. 大量產生資料 
dd_result = dd_app.create(builder, num_records=10, dataset_name="tutorial-101-dataset")

# 6. 評估您的資料品質
dd_analysis = dd_result.load_analysis()

# 生成分析報告
dd_analysis.to_report()