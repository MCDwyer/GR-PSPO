from datasets import Dataset, load_from_disk, DatasetDict

def format_dataset(dataset, tokenizer=None, for_test=False):
    formatted_data = []

    for entry in dataset:
        if 'question_concat' in entry:
            # for svamp
            content = entry['question_concat']
        elif 'question' in entry:
            # for gsmk8
            content = entry['question']

            if 'body' in entry:
                # for asdiv - need to combine the body and question column data
                content = entry['body'] + " " + content
        else:
            # for math-500 and competition_math
            content = entry['problem']
    
        conversation = [{'role': 'system', 
                        "content": "You are a careful math solver. Think through the solution and show the steps. Use English only. End the response with the final answer only in the format: '#### <final numeric answer only>'"},
                        {'role': 'user', 
                        'content': content}]
    
        if 'answer' in entry:
            # for gsmk8 and asdiv
            gold_ans = entry['answer']
        elif 'Answer' in entry:
            # for svamp
            gold_ans = entry['Answer']
        else:
            # for math-500 and competition_math
            gold_ans = entry['solution']

        formatted_entry = {'conversation': conversation, 
                            'gold_answer': gold_ans,}

        if for_test:
            if 'solution_type' in entry:
                # asdiv
                category = entry['solution_type']
                formatted_entry['category'] = category
            elif 'Type' in entry:
                # svamp
                category = entry['Type']
                formatted_entry['category'] = category
            elif 'subject' in entry:
                # math-500
                category = entry['subject']
                formatted_entry['category'] = category
            elif 'type' in entry:
                # competition_math
                category = entry['type']
                formatted_entry['category'] = category

            if 'level' in entry:
                # for math-500 and competition_math
                formatted_entry['difficulty'] = entry['level']

        if tokenizer is not None:
            formatted_entry['prompt'] = tokenizer.apply_chat_template(
                                        formatted_entry['conversation'],
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )

        formatted_data.append(formatted_entry)

    formatted_data = Dataset.from_list(formatted_data)

    return formatted_data

def sft_format(ds, tokenizer):
    formatted_data = []

    for entry in ds:
        conversation = [{'role': 'system', 
                        "content": "You are a careful math solver. Think through the solution and show the steps. Use English only. End the response with the final answer only in the format: '#### <final numeric answer only>'"},
            {'role': 'user', 'content': entry['question']},
            {'role': 'assistant', 'content': entry['answer']}]

        formatted_entry = {'conversation': conversation,}

        formatted_data.append(formatted_entry)

    formatted_data = Dataset.from_list(formatted_data)

    return formatted_data


def load_data(dataset_path, tokenizer=None, split_train=7000, seed=42, sft=False):
    ds = load_from_disk(dataset_path)

    all_train = ds["train"].shuffle(seed=seed)

    if sft:
        all_train = sft_format(all_train, tokenizer)
    else:
        all_train = format_dataset(all_train, tokenizer)

    train = all_train.select(range(split_train))
    val   = all_train.select(range(split_train, len(all_train) - 1))

    return DatasetDict({"train": train, "validation": val})

def load_test_data(dataset_path, seed=42):
    ds = load_from_disk(dataset_path)

    if "test" not in ds:
        if "validation" in ds:
            all_test = ds["validation"].shuffle(seed=seed)
        else:
            all_test = ds["train"].shuffle(seed=seed)
    else:
        all_test = ds["test"].shuffle(seed=seed)
    all_test = format_dataset(all_test, for_test=True)

    return all_test