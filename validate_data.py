import argparse
import json
from typing import Any, Dict, List


ALLOWED_ROLES = {"system", "user", "assistant"}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {i} 行 JSON 格式错误: {exc}") from exc
    return items


def validate_sample(sample: Dict[str, Any], index: int) -> List[str]:
    errors: List[str] = []
    required_fields = ["id", "style", "style_intensity", "messages"]
    for field in required_fields:
        if field not in sample:
            errors.append(f"样本#{index} 缺少字段: {field}")

    if "style" in sample and sample["style"] != "zhenhuan":
        errors.append(f"样本#{index} style 必须为 zhenhuan")

    if "style_intensity" in sample:
        si = sample["style_intensity"]
        if not isinstance(si, int) or si < 1 or si > 5:
            errors.append(f"样本#{index} style_intensity 必须是 1~5 的整数")

    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"样本#{index} messages 必须是长度>=2 的数组")
        return errors

    for m_idx, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            errors.append(f"样本#{index} message#{m_idx} 必须是对象")
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(f"样本#{index} message#{m_idx} role 非法: {role}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"样本#{index} message#{m_idx} content 不能为空字符串")

    if messages and messages[-1].get("role") != "assistant":
        errors.append(f"样本#{index} 最后一条消息建议为 assistant")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="校验甄嬛风格微调 JSONL 数据")
    parser.add_argument("--file", type=str, required=True, help="JSONL 文件路径")
    args = parser.parse_args()

    samples = read_jsonl(args.file)
    if not samples:
        print("数据为空")
        raise SystemExit(1)

    all_errors: List[str] = []
    for idx, sample in enumerate(samples, start=1):
        all_errors.extend(validate_sample(sample, idx))

    if all_errors:
        print(f"校验失败，共 {len(all_errors)} 个问题:")
        for err in all_errors[:100]:
            print(f"- {err}")
        if len(all_errors) > 100:
            print(f"... 其余 {len(all_errors) - 100} 个问题省略")
        raise SystemExit(1)

    print(f"校验通过，共 {len(samples)} 条样本。")


if __name__ == "__main__":
    main()
