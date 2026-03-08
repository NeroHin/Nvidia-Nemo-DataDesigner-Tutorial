# Nvidia NeMo DataDesigner 教學範例

本專案為 **NVIDIA NeMo DataDesigner** 的教學範例集，示範如何使用 NeMo Data Designer 進行高品質合成資料（Synthetic Data）生成，涵蓋入門教學與多種實務應用情境。

---

## 背景說明

### 什麼是 NeMo Data Designer？

NeMo Data Designer 是 NVIDIA 提供的**合成資料生成（Synthetic Data Generation, SDG）** 協調框架，透過大型語言模型（LLM）端點（如 NVIDIA、OpenAI、vLLM 等）產生高品質、可重現的合成資料集。

### 為什麼需要合成資料？

- **AI 模型訓練**：在缺乏真實資料或資料不足時，以合成資料進行訓練與微調
- **資料擴充**：增加資料多樣性，提升模型泛化能力
- **隱私保護**：避免使用含個人可識別資訊（PII）的真實資料
- **測試與評估**：產生邊界情境、壓力測試等特定用途的測試資料

### 本專案特色

- 以**繁體中文（台灣用語）**為主的範例與種子資料
- 涵蓋**醫療、金融、客服**等多領域應用
- 包含**預覽、生成、評估**的完整工作流程
- 示範 **LLM-as-Judge** 品質評估機制

---

## 環境需求

- Python 3.10+
- NVIDIA API Key（可至 [build.nvidia.com](https://build.nvidia.com) 取得）

---

## 安裝方式

### 1. 複製專案

```bash
git clone https://github.com/your-username/Nvidia-Nemo-DataDesigner-Tutorial.git
cd Nvidia-Nemo-DataDesigner-Tutorial
```

### 2. 建立虛擬環境（建議）

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# 或
venv\Scripts\activate      # Windows
```

### 3. 安裝套件

```bash
pip install data-designer python-dotenv
```

### 4. 設定 API 金鑰

複製 `.env.example` 為 `.env`，並填入您的 NVIDIA API 金鑰：

```bash
cp .env.example .env
```

編輯 `.env`：

```
NVIDIA_API_KEY=<您的_NVIDIA_API_金鑰>
```

> 若使用 OpenAI 或 OpenRouter，可改為設定 `OPENAI_API_KEY` 或 `OPENROUTER_API_KEY`。

### 5. 驗證設定

```bash
data-designer config list
```

若顯示已設定的模型提供者與模型清單，即表示設定完成。

---

## 專案結構

```
Nvidia-Nemo-DataDesigner-Tutorial/
├── README.md
├── .env.example              # 環境變數範例
├── data/                     # 種子資料
│   ├── data_person_1000_zh.json    # 個人資料（含醫療、金融欄位）
│   ├── qna_dataset_seed.json       # 客服 Q&A 種子
│   └── research_report_seed.json   # 研究報告種子
├── example/                  # 範例腳本
│   ├── data_designer_101.py        # 入門教學
│   ├── pii_dataset_generate.py    # PII 資料集生成
│   ├── qna_dataset_generate.py    # 客服 Q&A 資料集生成
│   └── research_report_generate.py # 研究報告生成
└── artifacts/                # 輸出結果（執行後產生）
```

---

## 範例說明

### 1. 入門教學：`data_designer_101.py`

示範 Data Designer 的基本工作流程：

1. 設定用於合成資料生成的模型
2. 載入種子資料集與欄位
3. 以提示詞與結構化輸出設定 LLM 欄位
4. 預覽資料集並依需求調整
5. 大量產生資料
6. 評估資料品質

**執行方式：**

```bash
cd example
python data_designer_101.py
```

### 2. PII 資料集生成：`pii_dataset_generate.py`

產生含個人可識別資訊（PII）與脫敏版本的成對資料集，適用於：

- Guardrails 偵測模型的正向／負向測試樣本
- PII 標註與評估（Ground Truth）

**流程：**

- Stage 1：生成含 PII 的長篇真實文件
- Stage 2：生成脫敏版本（non-PII）
- Stage 3：生成 PII Ground Truth 標註
- 品質評估：以 LLM-as-Judge 評分

### 3. 客服 Q&A 資料集：`qna_dataset_generate.py`

產生客服情境的問答對，支援：

- 多種使用者 persona、意圖、語氣
- 邊界情境（edge case）問句
- 政策約束下的標準回答
- 問答品質評估

### 4. 研究報告生成：`research_report_generate.py`

根據指定主題、讀者、風格等條件，產生結構化研究型文章：

- 先產出文章大綱（outline）
- 再依大綱擴充為完整報告
- 支援指定章節與關鍵字

---

## 使用方式

1. 進入 `example` 目錄
2. 確認 `data/` 路徑正確（範例中多使用 `./data/` 相對路徑）
3. 執行欲使用的範例腳本

```bash
cd example
python data_designer_101.py
# 或
python pii_dataset_generate.py
python qna_dataset_generate.py
python research_report_generate.py
```

輸出會儲存於 `artifacts/` 目錄，包含 Parquet 檔案與分析報告。

---

## 參考資源

- [NeMo Data Designer 官方文件](https://nvidia-nemo.github.io/DataDesigner/latest/)
- [NVIDIA NeMo DataDesigner GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)
- [NVIDIA Build 平台](https://build.nvidia.com)（取得 API 金鑰）
