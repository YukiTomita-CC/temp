import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = f"models/karakuri"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = """# 指示:

以下に「会話テーマ」とそれに対応する「会話データ」のペアをいくつか示します。その後、新しい「会話テーマ」を提示しますので、それに基づいて同じ形式で「会話データ」を作成してください。

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

**会話テーマ:** 睡眠時間どれくらい取れてる？

**会話データ:**

---

"""


    messages = [
        {"role": "user", "content": prompt}
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

    for i, output_id in enumerate(output_ids):
        print(f"Generating output {i}...")
        text = tokenizer.decode(output_id[input_ids.shape[-1]:], skip_special_tokens=True)

        import os
        os.makedirs(f"outputs/karakuri", exist_ok=True)

        with open(f"outputs/karakuri/output_1{i}.txt", "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
