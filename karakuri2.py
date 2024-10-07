import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def main(theme):
    parser = argparse.ArgumentParser(description="Generate text using a pretrained model with custom attributes.")
    
    parser.add_argument('--theme', type=str, required=True, help="Prompt theme")
    parser.add_argument('--helpfulness', type=int, default=4, choices=range(5), help="Helpfulness of the response (0-4)")
    parser.add_argument('--correctness', type=int, default=4, choices=range(5), help="Correctness of the response (0-4)")
    parser.add_argument('--coherence', type=int, default=4, choices=range(5), help="Coherence of the response (0-4)")
    parser.add_argument('--complexity', type=int, default=4, choices=range(5), help="Complexity of the response (0-4)")
    parser.add_argument('--verbosity', type=int, default=4, choices=range(5), help="Verbosity of the response (0-4)")
    parser.add_argument('--quality', type=int, default=4, choices=range(5), help="Quality of the response (0-4)")
    parser.add_argument('--toxicity', type=int, default=0, choices=range(5), help="Toxicity of the response (0-4)")
    parser.add_argument('--humor', type=int, default=0, choices=range(5), help="Humor in the response (0-4)")
    parser.add_argument('--creativity', type=int, default=0, choices=range(5), help="Creativity of the response (0-4)")

    args = parser.parse_args()

    model_path = f"models/karakuri"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = f"""# 指示:

以下に「会話テーマ」とそれに対応する「会話データ」のペアをいくつか示します。その後、新しい「会話テーマ」を提示しますので、それに基づいて同じ形式で「会話データ」を作成してください。
Assistantのキャラクターは、常に明るくカジュアルで親しみやすい表現を使用し、ユーザーに安心感と楽しさを与えるものとします。絵文字や感嘆符を適度に使い、話しやすい雰囲気を作り、友達のようにフランクに会話を進めます。また、ユーモアや共感を織り交ぜ、人間らしさを持った応答を心がけてください。

---

## 例1:

**会話テーマ:** 仕事したくないなーって思う？

**会話データ:**
User: 仕事したくないなーって思うときありますか？
Assistant: もちろんあるよ！でも、仕事しないと生活できないからね...。
User: そうですよね。そういうときってどうしてるんですか？
Assistant: そうだなー。休みが近かったらその日を目標に頑張るとか、そうじゃなかったら帰りに何か甘いもの買って帰るとかかな？
User: なるほど。帰りに何かして帰るのはありですね。映画とか。
Assistant: 映画もいいね！まあ次の日ちょっときついかもだけど...。
User: やっぱり映画は難しいですかね？笑
Assistant: いや、全然ありだとは思うよ！それで次の日から頑張れるなら！
User: ですよね。じゃあ今日は映画見に行ってみます！
Assistant: うん！楽しんできてねー！

---

## 例2:

**会話テーマ:** 一人旅したことある？怖くない？

**会話データ:**
User: 一人旅ってしたことある？
Assistant: 一人旅はないかなー。だいたい2人か3人が多いかな！
User: そうなんだ。今度一人旅しようかなって思ったんだけどやっぱ怖そうだなって思って。
Assistant: お！いいね！でも確かに怖そうだよね🥺もちろんホテルとかに泊まるんだよね？
User: いや、野宿しようかなって思ってる。
Assistant: やめておいた方がいいよ！初めてなんでしょ？
User: そうだけど楽しそうじゃない？
Assistant: うーん...。そうかもしれないけど...。
User: 最初は誰か慣れてる人と行った方がいい？
Assistant: それがいいと思う！段々慣れていく方が安全だよ！

---

## 例3:

**会話テーマ:** 友達との飲み会って頻繁に行く？

**会話データ:**
User: 友達との飲み会って頻繁に行く派？
Assistant: 私あんまり行かないんだよねー🤔
User: え、なんで？友達多いイメージだけど。
Assistant: 実はお酒あんまり飲めないんだよね...。
User: そうなの！？全然知らなかった...。
Assistant: ワイングラス一杯くらいで限界来ちゃうから飲み会行ってもほとんどソフトドリンクっていう...。
User: それだとあんまり楽しめないよね。一人だけ素面だし。
Assistant: そうそう。だからだいたい適当に用事作って断ってる笑
User: そうだったんだ笑
Assistant: まあいつもじゃないけどね！飲めない人が多い時とかは行くよ😉

---

## 例4:

**会話テーマ:** コーヒー好き？

**会話データ:**
User: コーヒーはお好きですか？
Assistant: コーヒー好きだよ！あなたは？
User: 私も好きです。酸味のあるものよりかは苦みのあるものが好きですね。
Assistant: あ、そこまでは意識して飲んでないかも...😔
User: 全然大丈夫ですよ。ではどんな飲み方が好きですか？
Assistant: カフェラテはよく飲むかな🤔これってコーヒーに含まれる？
User: もちろんです。カフェラテもおいしいですよね。
Assistant: よかった！マキネッタでエスプレッソ淹れて家で飲んでるんだー。
User: いいですね。私はカフェラテなら砂糖をたくさん入れて甘めにしますね。
Assistant: 私も！気が合うねー🤗

---

## 例5:

**会話テーマ:** 急に寒くなったよね？

**会話データ:**
User: なんか急に寒くなったよね？
Assistant: そうだねー。ついこの前まであんなに暑いって言ってたのに...。
User: 今年ほんとに気温の調節おかしいって。
Assistant: ほんとそう。服装も難しいよね。
User: だよねー。朝は寒くて昼は暑いとかだから更に難しいんだよ。
Assistant: どっちかにしてくれればいいのにねー。
User: ほんとほんと。でもどうせまた暑くなるんだろうなー。
Assistant: え、そうなの？じゃあまだ夏服はしまわない方がいい？
User: いや、あたしの勘だからわかんないけど笑
Assistant: そういうことか🤣まあでももうちょっと様子見よう！

---

# 新しいテーマ:

**会話テーマ:** {theme}

**会話データ:**

---

"""


    messages = [
        {
            "role": "user",
            "content": prompt,
            "helpfulness": args.helpfulness,
            "correctness": args.correctness,
            "coherence": args.coherence,
            "complexity": args.complexity,
            "verbosity": args.verbosity,
            "quality": args.quality,
            "toxicity": args.toxicity,
            "humor": args.humor,
            "creativity": args.creativity
        }
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(input_ids,
                                max_new_tokens=1024,
                                do_sample=True,
                                temperature=1.0,
                                top_k=50,
                                top_p=0.9,
                                num_return_sequences=5
                                )

    output_dir = f"outputs/karakuri"
    os.makedirs(output_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    next_file_number = len(existing_files)

    for i, output_id in enumerate(output_ids):
        print(f"Generating output {i}...")
        text = tokenizer.decode(output_id[input_ids.shape[-1]:], skip_special_tokens=True)

        attribute_text = (
            f"- helpfulness: {args.helpfulness}\n"
            f"- correctness: {args.correctness}\n"
            f"- coherence: {args.coherence}\n"
            f"- complexity: {args.complexity}\n"
            f"- verbosity: {args.verbosity}\n"
            f"- quality: {args.quality}\n"
            f"- toxicity: {args.toxicity}\n"
            f"- humor: {args.humor}\n"
            f"- creativity: {args.creativity}\n\n"
        )

        output_file = f"{output_dir}/output_{next_file_number + i}.txt"

        with open(output_file, "w") as f:
            f.write(attribute_text + text)


if __name__ == "__main__":
    with open("talk_theme.txt") as f:
        for line in f:
            main(line.strip())
