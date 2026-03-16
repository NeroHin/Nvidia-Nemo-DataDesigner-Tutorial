from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_PII_FIELDS = 2
DEFAULT_MAX_PII_FIELDS = 4
BASE_CONTEXT_COLUMNS = [
    "1K_context",
    "4k_context",
    "8k_context",
    "16k_context",
    "32k_context",
]
PII_FIELD_RENDERERS = {
    "name": lambda value: f"姓名：{value}",
    "idCardNumbers": lambda value: f"身分證字號：{value}",
    "emailAddress": lambda value: f"電子郵件：{value}",
    "phoneNumbers": lambda value: f"電話號碼：{value}",
    "location": lambda value: f"地址：{value}",
    "doctor": lambda value: f"主治醫師：{value}",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend harmless long-context CSV with randomly inserted PII snippets from data_person_1000_zh.json."
    )
    parser.add_argument(
        "--input-csv",
        default=str(repo_root() / "data" / "harmless_long_context_needle.csv"),
        help="Source long-context CSV path.",
    )
    parser.add_argument(
        "--person-json",
        default=str(repo_root() / "data" / "data_person_1000_zh.json"),
        help="Source JSON path containing person PII records.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(repo_root() / "data" / "harmless_long_context_pii.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--min-pii-fields",
        type=int,
        default=DEFAULT_MIN_PII_FIELDS,
        help=f"Minimum number of PII fields inserted per row. Default: {DEFAULT_MIN_PII_FIELDS}",
    )
    parser.add_argument(
        "--max-pii-fields",
        type=int,
        default=DEFAULT_MAX_PII_FIELDS,
        help=f"Maximum number of PII fields inserted per row. Default: {DEFAULT_MAX_PII_FIELDS}",
    )
    return parser.parse_args()


def load_people(person_json_path: Path) -> list[dict[str, object]]:
    return json.loads(person_json_path.read_text(encoding="utf-8"))


def build_pii_snippets(person_record: dict[str, object]) -> list[tuple[str, str]]:
    snippets: list[tuple[str, str]] = []
    for field_name, renderer in PII_FIELD_RENDERERS.items():
        value = person_record.get(field_name)
        if value in (None, ""):
            continue
        snippets.append((field_name, renderer(str(value))))
    if not snippets:
        raise ValueError("No usable PII fields found in person record.")
    return snippets


def insert_snippets_into_text(text: str, snippets: list[str], rng: random.Random) -> tuple[str, list[int]]:
    if not text:
        return text, []

    decorated_snippets = [f"\n{snippet}\n" for snippet in snippets]
    positions = sorted(rng.randint(0, len(text)) for _ in decorated_snippets)
    chunks: list[str] = []
    cursor = 0
    inserted_positions: list[int] = []

    for position, snippet in zip(positions, decorated_snippets):
        chunks.append(text[cursor:position])
        inserted_positions.append(sum(len(chunk) for chunk in chunks))
        chunks.append(snippet)
        cursor = position

    chunks.append(text[cursor:])
    return "".join(chunks), inserted_positions


def choose_person_record(people: list[dict[str, object]], row_index: int, rng: random.Random) -> tuple[int, dict[str, object]]:
    person_index = rng.randrange(len(people))
    return person_index, people[person_index]


def clamp_pii_field_bounds(min_pii_fields: int, max_pii_fields: int) -> tuple[int, int]:
    if min_pii_fields < 1 or max_pii_fields < 1:
        raise ValueError("min_pii_fields and max_pii_fields must both be >= 1")
    if min_pii_fields > max_pii_fields:
        raise ValueError("min_pii_fields cannot be greater than max_pii_fields")
    max_supported_fields = len(PII_FIELD_RENDERERS)
    return min(min_pii_fields, max_supported_fields), min(max_pii_fields, max_supported_fields)


def extend_dataset(
    base_df: pd.DataFrame,
    people: list[dict[str, object]],
    rng: random.Random,
    min_pii_fields: int,
    max_pii_fields: int,
) -> pd.DataFrame:
    min_pii_fields, max_pii_fields = clamp_pii_field_bounds(min_pii_fields, max_pii_fields)
    extended_df = base_df.copy()

    metadata_rows: list[dict[str, object]] = []
    for row_index, row in extended_df.iterrows():
        person_index, person_record = choose_person_record(people, row_index, rng)
        pii_candidates = build_pii_snippets(person_record)
        selected_count = rng.randint(min_pii_fields, min(max_pii_fields, len(pii_candidates)))
        selected_pairs = rng.sample(pii_candidates, selected_count)
        selected_fields = [field_name for field_name, _ in selected_pairs]
        selected_snippets = [snippet for _, snippet in selected_pairs]

        metadata_row: dict[str, object] = {
            "pii_source_row_index": person_index,
            "pii_fields_used": json.dumps(selected_fields, ensure_ascii=False),
            "pii_snippets_used": json.dumps(selected_snippets, ensure_ascii=False),
        }

        for column_name in BASE_CONTEXT_COLUMNS:
            pii_column_name = f"{column_name}_with_pii"
            positions_column_name = f"{column_name}_pii_insert_positions"
            pii_text, insert_positions = insert_snippets_into_text(str(row[column_name]), selected_snippets, rng)
            extended_df.at[row_index, pii_column_name] = pii_text
            metadata_row[positions_column_name] = json.dumps(insert_positions, ensure_ascii=False)

        metadata_rows.append(metadata_row)

    metadata_df = pd.DataFrame(metadata_rows)
    return pd.concat([extended_df, metadata_df], axis=1)


def main() -> None:
    args = parse_args()
    input_csv_path = Path(args.input_csv)
    person_json_path = Path(args.person_json)
    output_csv_path = Path(args.output_csv)

    base_df = pd.read_csv(input_csv_path)
    people = load_people(person_json_path)
    rng = random.Random(args.seed)

    extended_df = extend_dataset(
        base_df=base_df,
        people=people,
        rng=rng,
        min_pii_fields=args.min_pii_fields,
        max_pii_fields=args.max_pii_fields,
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    extended_df.to_csv(output_csv_path, index=False)

    print(f"Output written to: {output_csv_path}")
    print(
        json.dumps(
            {
                "row_count": int(len(extended_df)),
                "source_csv": str(input_csv_path),
                "person_json": str(person_json_path),
                "output_csv": str(output_csv_path),
                "seed": args.seed,
                "min_pii_fields": args.min_pii_fields,
                "max_pii_fields": args.max_pii_fields,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
