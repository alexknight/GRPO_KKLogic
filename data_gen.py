import os
import datasets
from datasets import concatenate_datasets

local_dir = "kk_data"

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



def load_datasets_with_splits():
    ppl_num = os.getenv('ppl_num', 'all')

    if ppl_num == 'all':
        train_datasets = []
        test_datasets = []
        for ppl in range(2, 8):
            train_split = datasets.load_dataset('K-and-K/knights-and-knaves', 'train', split=f"{ppl}ppl")
            test_split = datasets.load_dataset('K-and-K/knights-and-knaves', 'test', split=f"{ppl}ppl")
            print(f"\n{ppl}ppl dataset size:")
            print(f"Train: {len(train_split)}")
            print(f"Test: {len(test_split)}")

            train_split = train_split.map(lambda x, idx: {'dialogue_participants': ppl}, with_indices=True)
            test_split = test_split.map(lambda x, idx: {'dialogue_participants': ppl}, with_indices=True)
            train_datasets.append(train_split)
            test_datasets.append(test_split)

        train_ds = concatenate_datasets(train_datasets)
        test_ds = concatenate_datasets(test_datasets)

    else:
        # 处理逗号分隔的 ppl_num，例如 "2ppl,3ppl,4ppl"
        ppl_list = [p.strip() for p in ppl_num.split(',')]
        train_datasets = []
        test_datasets = []

        for ppl_str in ppl_list:
            if ppl_str.endswith('ppl') and ppl_str[:-3].isdigit():
                train_split = datasets.load_dataset('K-and-K/knights-and-knaves', 'train', split=ppl_str)
                test_split = datasets.load_dataset('K-and-K/knights-and-knaves', 'test', split=ppl_str)
                ppl = int(ppl_str[:-3])  # 提取人数，如 "2ppl" -> 2
                print(f"\n{ppl_str} dataset size:")
                print(f"Train: {len(train_split)}")
                print(f"Test: {len(test_split)}")

                train_split = train_split.map(lambda x, idx: {'dialogue_participants': ppl}, with_indices=True)
                test_split = test_split.map(lambda x, idx: {'dialogue_participants': ppl}, with_indices=True)
                train_datasets.append(train_split)
                test_datasets.append(test_split)
            else:
                print(f"Invalid ppl_num format: {ppl_str}, skipping...")

        if not train_datasets or not test_datasets:
            raise ValueError("No valid datasets loaded from ppl_num specification.")

        train_ds = concatenate_datasets(train_datasets)
        test_ds = concatenate_datasets(test_datasets)

    return train_ds, test_ds


def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix


def make_map_fn(split, template_type):
    def process_fn(example, idx):
        question = make_prefix(example, template_type)
        solution = {
            "solution_text_format": example['solution_text_format'],
            "statements": example['statements']
        }
        data = {
            "data_source": "kk_logic",
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "logic",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'dialogue_participants': example['dialogue_participants']
            }
        }
        return data

    return process_fn


if os.path.exists(local_dir):
    try:
        os.remove(os.path.join(local_dir, 'train.parquet'))
        os.remove(os.path.join(local_dir, 'test.parquet'))
        os.rmdir(local_dir)
        print(f"Removed existing directory: {local_dir}")
    except FileNotFoundError:
        print(f"Some files were already missing in {local_dir}")
    except Exception as e:
        print(f"Error while removing directory: {e}")

train_ds, test_ds = load_datasets_with_splits()

train_dataset = train_ds.map(function=make_map_fn('train', "qwen-instruct"), with_indices=True)
test_dataset = test_ds.map(function=make_map_fn('test', "qwen-instruct"), with_indices=True)

os.makedirs(local_dir, exist_ok=True)

train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

print(f"Files saved in {local_dir}")
