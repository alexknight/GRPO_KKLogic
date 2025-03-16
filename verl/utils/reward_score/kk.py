import re
from typing import Tuple, Optional
import os

import re
from typing import Dict


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.

    Args:
        solution_text: Formatted solution text from dataset

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b',
            re.IGNORECASE
        )
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def validate_response_structure_extended(processed_str: str) -> str:
    """扩展的格式验证函数，返回"correct"、"partial"或"incorrect"
    - correct: 完全符合格式要求
    - partial: 基本结构正确但有小问题
    - incorrect: 严重的格式问题
    """
    # 基本格式检查
    basic_format_correct = validate_response_structure(processed_str)

    if not basic_format_correct:
        return "incorrect"

    # 检查是否有细节问题
    has_minor_issues = False

    # 示例检查点（根据具体任务定制）:
    # 1. 检查标签是否完整闭合
    unclosed_tags = check_unclosed_tags(processed_str)
    # 2. 检查必要部分是否存在（如答案区域）
    missing_sections = check_missing_sections(processed_str)

    if unclosed_tags or missing_sections:
        has_minor_issues = True

    return "partial" if has_minor_issues else "correct"


def compute_score(solution_str: str,
                  ground_truth: dict[str, str],
                  format_reward: int = 1,
                  answer_reward: float = 1.0) -> float:
    """Computes score for model response."""
    print("solution_str=", solution_str)

    # 从环境变量获取阶段
    stage = os.getenv('TRAINING_STAGE', 'stage1').lower()

    # 使用传入的参数设置基本权重
    format_weight = format_reward  # 使用传入的format_reward参数
    answer_weight = answer_reward  # 使用传入的answer_reward参数

    # 从环境变量获取推理奖励权重
    reasoning_weight = float(os.getenv('REASONING_BONUS_BASE', '0.0' if stage == 'stage1' else '0.5'))

    # 提取答案和处理后的字符串
    answer_text, processed_str = extract_solution(solution_str)

    # 1. 格式验证 - 增加部分正确的状态
    format_status = validate_response_structure_extended(processed_str)
    if format_status == "correct":
        format_score = format_weight
    elif format_status == "partial":
        format_score = 0.5 * format_weight
    else:  # "incorrect"
        format_score = -format_weight

    # 2. 答案验证 - 只在格式至少部分正确时计算
    answer_score = 0.0
    if answer_text and format_score > 0:
        solution_text = ground_truth.get('solution_text_format', '')
        gt_status = parse_solution_text_format(solution_text)
        expected_names = list(gt_status.keys())

        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            if pred_status == gt_status:  # 完全正确
                answer_score = answer_weight
            else:  # 部分正确
                match_count = sum(1 for k, v in pred_status.items()
                                  if k in gt_status and v == gt_status[k])
                match_ratio = match_count / len(gt_status)

                # 0.5以上的匹配被视为部分正确，获得部分分数
                if match_ratio >= 0.5:
                    answer_score = 0.5 * answer_weight
                else:
                    answer_score = -0.5 * answer_weight  # 小惩罚
        else:
            answer_score = -answer_weight  # 无法解析答案

    # 3. 推理评估 - 只在Stage2且格式和答案都至少部分正确时计算
    reasoning_score = 0.0
    if stage == "stage2" and format_score > 0 and answer_score > 0:
        reasoning_score = evaluate_reasoning(processed_str, ground_truth) * reasoning_weight

    # 总分计算
    if stage == "stage1":
        # Stage1: 主要看格式分
        total_score = format_score
    else:
        # Stage2: 格式分 + 答案分 + 推理分
        total_score = format_score + answer_score + reasoning_score

    return total_score




def evaluate_reasoning(processed_str: str, question_context: dict = None) -> float:
    """评估推理质量，返回0到1之间的得分，减少关键词依赖"""
    # 提取思考部分
    think_matches = re.findall(r'<think>(.*?)</think>', processed_str, re.DOTALL)
    if not think_matches:
        return 0.0

    think_content = think_matches[0].strip()

    # 基础检查 - 如果内容太短则直接返回低分
    if len(think_content) < 50:
        return 0.0

    # 1. 结构评分 (0.5分) - 检查推理是否有清晰的结构
    structure_score = 0.0

    # 检查是否有清晰的段落划分
    paragraphs = [p for p in think_content.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:  # 至少有两个段落
        structure_score += 0.2

    # 检查是否包含明确的步骤表示
    step_pattern = r'(step|第.步|首先|然后|接下来|最后|first|second|finally|1\.|2\.)'
    steps = re.findall(step_pattern, think_content.lower())
    if len(steps) >= 2:  # 至少包含两个步骤标记
        structure_score += 0.3

    # 2. 内容密度评分 (0.5分) - 检查是否包含实质内容而非填充
    content_score = 0.0

    # 计算数值、技术词汇、实体名称等信息密度
    # 数字计数
    number_count = len(re.findall(r'\d+', think_content))

    # 句子平均长度
    sentences = [s for s in re.split(r'[.!?。！？]', think_content) if s.strip()]
    avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)

    # 基于这些指标计算内容密度分数
    if number_count >= 3:  # 包含一定数量的数字信息
        content_score += 0.2

    if avg_sentence_len > 15:  # 句子平均长度足够表达复杂思想
        content_score += 0.3

    # 将题目信息与推理内容匹配度
    # 如果有问题上下文,可以检查推理中是否使用了问题中的关键信息
    if question_context:
        # 实现简化,实际使用时应根据question_context内容提取关键信息
        # 并检查这些信息是否出现在推理中
        pass

    # 总分计算 - 结构分 + 内容分
    total_score = structure_score + content_score

    return min(total_score, 1.0)  # 确保得分不超过1.0


# 辅助函数
def check_unclosed_tags(processed_str: str) -> bool:
    """检查未闭合的标签"""
    # 简化实现 - 可根据具体格式要求定制
    open_tags = re.findall(r'<(\w+)>', processed_str)
    close_tags = re.findall(r'</(\w+)>', processed_str)
    return len(open_tags) != len(close_tags)


def check_missing_sections(processed_str: str) -> bool:
    """检查是否缺少必要的部分"""
    # 简化实现 - 可根据具体格式要求定制
    required_sections = ['<answer>', '</answer>']
    return not all(section in processed_str for section in required_sections)


if __name__ == '__main__':
    os.environ['TRAINING_STAGE']   = 'stage2'
    compute_score(
        solution_str=''''<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\nA very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 3 inhabitants: Michael, Mason, and Jacob. Michael commented, "Mason is not a knight". Mason commented, "Michael is a knight and Jacob is a knave". Jacob stated, "If Michael is a knight then Jacob is a knight". So who is a knight and who is a knave?\n<|im_end|>\n<|im_start|>assistant\n<think><think> First, let's analyze Aria's statement: "If Michael is a knight then Michael is a knave". This is a contradiction since knights always tell the truth, so Michael cannot be a knight. If Michael is a knave, his statement is also a lie, meaning the conditional statement is true. Therefore, Aria is a knave. <answer> (1) Aria is a knight\n(2) Elizabeth is a knave </answer><|endoftext|>''',
        ground_truth={'solution_text_format': '(1)  Elizabeth is a knave\n(2) Aria is a knight',
   'statements': ""},
        format_reward=3.0,
        answer_reward=1.0
    )

