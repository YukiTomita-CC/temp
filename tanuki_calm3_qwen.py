import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_name):
    if model_name not in ["calm3", "tanuki", "qwen"]:
        print("model_name must be one of 'calm3', 'tanuki', 'qwen'.")
        return
    
    model_name = f"models/{model_name}"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    import json


    conversation_file_path = 'memo1.json'
    persona_list_file_path = 'user_persona.json'

    with open(conversation_file_path, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)

    with open(persona_list_file_path, 'r', encoding='utf-8') as f:
        persona_list = json.load(f)

    def generate_prompt(conversation_data, persona_list, target_persona_id):
        old_persona_id = conversation_data.get('user_persona_id')
        theme = conversation_data.get('theme')
        
        old_persona = next((p for p in persona_list if p['id'] == old_persona_id), None)
        new_persona = next((p for p in persona_list if p['id'] == target_persona_id), None)
        
        conversation_lines = []
        for turn in conversation_data['conversations']:
            role = 'User' if turn['role'] == 'user' else 'Assistant'
            content = turn['content']
            conversation_lines.append(f"{role}: {content}")
        conversation_text = '\n'.join(conversation_lines)
        
        def format_persona(persona, persona_name):
            items = [f"**{persona_name}の会話データのユーザーペルソナ**：\n"]
            for key, value in persona.items():
                if key != 'id':
                    items.append(f"- {key}: {value}")
            return '\n'.join(items)
        
        old_persona_text = format_persona(old_persona, "既存")
        new_persona_text = format_persona(new_persona, "新規")
        
        return conversation_text, old_persona_text, new_persona_text, old_persona.get('名前'), new_persona.get('名前')

    target_persona_id = 2
    conversation_text, old_persona_text, new_persona_text, old_persona_name, new_persona_name = generate_prompt(conversation_data, persona_list, target_persona_id)

    prompt = f"""**指示**：

    以下の既存の会話データは、ユーザーペルソナ「{old_persona_name}」によるものです。この会話を、新しいユーザーペルソナ「{new_persona_name}」に合わせて、ユーザーの発言内容と言葉遣いを調整してください。**アシスタントの発言も、会話の自然さを保つために必要な場合のみ最小限の変更を行ってください**。テーマはそのままにしてください。

    """

    prompt = prompt + f"""**既存の会話データ**：

    ```
    {conversation_text}
    ```

    """

    prompt = prompt + f"{old_persona_text}\n\n"
    prompt = prompt + f"{new_persona_text}\n\n"

    prompt = prompt + """**出力形式**：

    ```
    User: [ユーザーの発言]
    Assistant: [アシスタントの発言]
    ...
    ```

    **注意事項**：

    - ユーザーの一人称や言葉遣いを新しいペルソナに合わせて変更してください。
    - ユーザーの性格や趣味を反映し、発言内容を自然に調整してください。
    - **アシスタントの発言も、会話の自然さを保つために必要な場合のみ最小限の変更を行ってください。ただし、元の意図や情報は維持してください。**
    - 使用できる絵文字は次に挙げるものだけです。それ以外は使用しないでください。「😊,🥰,😉,🤗,😭,🤣,😆,🙄,😡,😲,😳,😔,😇,😙,🥳,🤔,🥺,🥱,💓,✨」
    """

    system_prompt = "あなたは親切なAIアシスタントです。"
    if model_name == "tanuki":
        system_prompt = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"

    temperature = 0.5
    if model_name == "qwen":
        temperature = 1.0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(input_ids,
                                max_new_tokens=1024,
                                do_sample=True,
                                temperature=temperature,
                                top_k=50,
                                top_p=0.9,
                                num_return_sequences=5
                                )

    for i, output_id in enumerate(output_ids):
        text = tokenizer.decode(output_id[input_ids.shape[-1]:], skip_special_tokens=True)

        with open(f"outputs/{model_name}/output_{i}.txt", "w") as f:
            f.write(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    main(args.model_name)
