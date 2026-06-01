#!/usr/bin/env python3
"""Evaluate fine-tuned CWS model against validated test cases.

Sweeps multiple OOV penalty values and compares fine-tuned model
against the baseline ckiplab/bert-base-chinese-ws model side-by-side.

All configuration is defined as file-level constants, no CLI args.
"""

from pathlib import Path
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

from cws import (
    load_cedict_vocab,
    get_begin_probs,
    segment_sentence,
    score_segmentation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent.parent

CHECKPOINT_PATH = str(ML_DIR / "models" / "cws_finetuned" / "final")
CEDICT_PATH = str(ML_DIR / "data" / "cedict_ts.u8")
BASELINE_MODEL = "ckiplab/bert-base-chinese-ws"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Validated test cases: (input_sentence, golden_segmentation)
# Golden uses dash-separated words.
# ---------------------------------------------------------------------------

VALIDATED_TEST_CASES = [
    # --- From ebooks ---
    ("倭寇侵华日，书生投笔时", "倭寇-侵华-日-，-书生-投-笔-时"),
    ("毁家纾国难，大义不容辞", "毁-家-纾-国难-，-大义-不容-辞"),
    ("她男友还好赌，在APP上买六合彩，性质跟赌博一样", "她-男友-还-好-赌-，-在-A-P-P-上-买-六合彩-，-性质-跟-赌博-一样"),

    # --- Sanity checks ---
    ("今天天气很好", "今天-天气-很-好"),
    ("我喜欢吃苹果", "我-喜欢-吃-苹果"),
    ("他是一个学生", "他-是-一个-学生"),
    ("他每天在办公室工作", "他-每天-在-办公室-工作"),
    ("他把手机放在桌上", "他-把-手机-放-在-桌-上"),

    # --- Literary & classical ---
    # Poetry
    ("芜湖如诗如画，青山环抱，江水悠悠", "芜湖-如诗如画-，-青山-环抱-，-江水-悠悠"),
    ("落霞与孤鹜齐飞，秋水共长天一色", "落-霞-与-孤-鹜-齐-飞-，-秋水-共-长-天-一-色"),
    ("山重水复疑无路，柳暗花明又一村。这首诗出自陆游的游山西村。", "山-重-水-复-疑-无-路-，-柳暗花明-又-一-村-。-这-首-诗-出自-陆游-的-游-山西-村-。"),
    ("人生若只如初见，何事秋风悲画扇", "人生-若-只-如初-见-，-何-事-秋-风-悲-画-扇"),
    ("曾经沧海难为水，除却巫山不是云", "曾经沧海-难为-水-，-除-却-巫山-不是-云"),
    ("此情可待成追忆，只是当时已惘然", "此-情-可-待-成-追忆-，-只是-当时-已-惘然"),
    ("床前明月光，疑是地上霜", "床-前-明月-光-，-疑-是-地上-霜"),
    ("举头望明月，低头思故乡", "举-头-望-明月-，-低头-思-故乡"),
    ("国破山河在，城春草木深", "国-破-山河-在-，-城-春-草木-深"),
    ("感时花溅泪，恨别鸟惊心", "感-时-花-溅-泪-，-恨-别-鸟-惊心"),
    ("大江东去，浪淘尽，千古风流人物", "大-江-东-去-，-浪-淘-尽-，-千古-风流-人物"),
    ("明月几时有，把酒问青天", "明月-几时-有-，-把酒-问-青天"),
    ("但愿人长久，千里共婵娟", "但愿-人-长久-，-千里-共-婵娟"),
    ("暮霭沉沉楚天阔", "暮霭-沉沉-楚-天-阔"),
    ("斜阳草树，寻常巷陌", "斜阳-草-树-，-寻常-巷-陌"),
    ("纵使相逢应不识，尘满面，鬓如霜", "纵使-相逢-应-不-识-，-尘-满面-，-鬓-如-霜"),
    ("寂寥无人，凄神寒骨", "寂寥-无人-，-凄-神-寒-骨"),
    ("风萧萧兮易水寒", "风-萧-萧-兮-易-水-寒"),
    # Classical prose
    ("逝者如斯夫，不舍昼夜", "逝者-如斯-夫-，-不舍-昼夜"),
    ("学而时习之，不亦说乎", "学-而-时-习-之-，-不亦-说-乎"),
    ("三人行，必有我师焉", "三人行-，-必-有-我-师-焉"),
    ("知之为知之，不知为不知，是知也", "知-之-为-知-之-，-不知-为-不知-，-是-知-也"),
    ("吾日三省吾身", "吾-日-三-省-吾-身"),
    # Rare/literary vocabulary
    ("他踽踽独行于荒野之中", "他-踽踽独行-于-荒野-之中"),
    ("她的眼眸中有一丝怅惘", "她-的-眼眸-中-有-一-丝-怅惘"),

    # --- Ambiguous boundaries & context-dependent ---
    # Structural ambiguity
    ("研究生命的起源", "研究-生命-的-起源"),
    ("下雨天留客天留我不留", "下雨-天-留客-天-留-我-不-留"),
    ("南京市长江大桥", "南京市-长江-大桥"),
    ("发展中国家", "发展中国家"),
    ("结合成分来看", "结合-成分-来-看"),
    # Context determines the split
    ("这个人有时候很好说话", "这个-人-有时候-很-好-说话"),
    ("他说的确实在理", "他-说-的-确实-在理"),
    ("他从马上下来", "他-从-马上-下来"),
    ("这里人才多", "这里-人才-多"),
    ("他有点儿意思", "他-有点儿-意思"),
    ("我想起来了", "我-想起来-了"),
    ("他们都过来了", "他们-都-过来-了"),
    ("这种做法不对头", "这种-做法-不对头"),
    ("他的话很难听", "他-的-话-很-难听"),
    ("这件事很难办", "这-件-事-很-难-办"),
    ("他马上就要上马了", "他-马上-就要-上马-了"),
    ("这件事情很难说清楚", "这-件-事情-很-难-说-清楚"),
    ("我们要为人民服务", "我们-要-为人民服务"),
    ("这本书的内容很有意思", "这-本-书-的-内容-很-有意思"),
    ("她对他的看法有所改变", "她-对-他-的-看法-有所-改变"),
    # Merge ambiguity — longer CEDICT word vs shorter split
    ("印巴两国关系紧张", "印巴-两国-关系-紧张"),
    ("请与我们同行", "请-与-我们-同行"),
    ("他是我的同行", "他-是-我-的-同行"),
    ("城管部门处处帮助她们", "城管-部门-处处-帮助-她们"),
    ("在北京打车很方便", "在-北京-打车-很-方便"),
    ("克林顿的中国之行会很成功", "克林顿-的-中国-之-行-会-很-成功"),
    ("代孕在很多国家是违法的", "代孕-在-很-多-国家-是-违法-的"),
    ("网络路由器的配置很复杂", "网络-路由器-的-配置-很-复杂"),
    ("这是一个军师级单位，下辖三个团", "这-是-一个-军-师-级-单位-，-下辖-三-个-团"),
    ("俗话说船大难掉头，大企业转型很慢", "俗话说-船-大-难-掉头-，-大-企业-转型-很-慢"),
    ("赤峰市是一个少林区，森林覆盖率很低", "赤峰市-是-一个-少-林-区-，-森林-覆盖率-很-低"),
    ("孩子们成了爷爷奶奶的座上客", "孩子们-成-了-爷爷-奶奶-的-座上客"),
    ("有的是蔡老的同辈", "有的-是-蔡-老-的-同辈"),
    ("他担任副主席一职", "他-担任-副主席-一-职"),
    ("海关加强了反走私力度", "海关-加强-了-反走私-力度"),
    ("花园里开满了百合花", "花园-里-开满-了-百合花"),
    # Colloquial
    ("你这人怎么这样啊", "你-这-人-怎么-这样-啊"),
    ("他这话说得也太绝了", "他-这-话-说-得-也-太-绝-了"),
    ("这事儿办得不地道", "这-事儿-办-得-不-地道"),

    # --- Named entities ---
    ("习近平主席访问了白宫", "习近平-主席-访问-了-白宫"),
    ("张爱玲的小说倾城之恋很有名", "张爱玲-的-小说-倾城-之-恋-很-有名"),
    ("清华大学计算机系的研究成果", "清华大学-计算机-系-的-研究-成果"),
    ("在北京大学未名湖畔散步", "在-北京大学-未名-湖-畔-散步"),
    ("我来自中华人民共和国", "我-来自-中华人民共和国"),
    ("鲁迅原名周树人", "鲁迅-原名-周树人"),
    ("李白杜甫并称李杜", "李白-杜甫-并称-李-杜"),

    # --- Idioms & set phrases ---
    ("他一意孤行不听劝告", "他-一意孤行-不-听-劝告"),
    ("这件事扑朔迷离难以判断", "这-件-事-扑朔迷离-难以-判断"),
    ("他们之间的关系藕断丝连", "他们-之间-的-关系-藕断丝连"),
    ("事情的发展出人意料", "事情-的-发展-出人意料"),
    ("他做事总是虎头蛇尾", "他-做事-总是-虎头蛇尾"),
    ("他对这件事心知肚明", "他-对-这-件-事-心知肚明"),
    ("这个计划胎死腹中", "这个-计划-胎死腹中"),
    ("他说话总是拐弯抹角", "他-说话-总是-拐弯抹角"),

    # --- Domain-specific & technical ---
    ("量子计算机的并行处理能力", "量子-计算机-的-并行-处理能力"),
    ("深度学习模型的反向传播算法", "深度学习-模型-的-反向-传播-算法"),
    ("自然语言处理中的分词问题", "自然语言处理-中-的-分词-问题"),
    ("卷积神经网络的特征提取层", "卷-积-神经网络-的-特征-提取-层"),

    # --- Long & complex ---
    ("在这个充满不确定性的时代里我们更需要保持内心的平静与坚定", "在-这个-充满-不-确定性-的-时代-里-我们-更-需要-保持-内心-的-平静-与-坚定"),
    ("他虽然表面上看起来若无其事但内心深处却波涛汹涌", "他-虽然-表面上-看起来-若无其事-但-内心深处-却-波涛汹涌"),
    ("那些曾经以为刻骨铭心的记忆终究会随着时间的流逝而渐渐淡去", "那些-曾经-以为-刻骨铭心-的-记忆-终究-会-随着-时间-的-流逝-而-渐渐-淡-去"),

    # --- LLM annotation edge cases (from dataset_v2 testing) ---
    ("拉马来，我去回太爷去", "拉-马-来-，-我-去-回-太爷-去"),
    ("谁不是袭人拿下马来的", "谁-不是-袭-人-拿-下马-来-的"),
    ("我也赏鉴赏鉴", "我-也-赏鉴-赏鉴"),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str):
    """Load a BertForTokenClassification model + tokenizer."""
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_eval(
    models: list[tuple[str, object, object]],
    cedict_vocab: set[str],
    test_cases: list[tuple[str, str]],
    device: str,
):
    """
    Run evaluation across models.

    models: list of (name, tokenizer, model)
    """
    total = len(test_cases)
    tallies = {name: {"perfect": 0, "oversplit": 0, "wrong": 0} for name, _, _ in models}

    for i, (sentence, gold) in enumerate(test_cases, 1):
        print(f"\n[{i:02d}] Input: {sentence}")
        print(f"     Gold:  {gold}")

        any_wrong = False
        for name, tokenizer, model in models:
            result = segment_sentence(
                sentence, tokenizer, model,
                device=device,
                cedict_vocab=cedict_vocab,
            )
            score = score_segmentation(result, gold)
            tallies[name][score] += 1
            mark = {"perfect": "✓✓", "oversplit": "✓~", "wrong": "✗"}[score]
            print(f"     {name}: {result} {mark}")
            if score == "wrong":
                any_wrong = True

        if any_wrong:
            print("     --- P(B) per char ---")
            for name, tokenizer, model in models:
                chars, probs = get_begin_probs(sentence, tokenizer, model, device=device)
                print(f"     {name}: {' | '.join(f'{c}:{p:.2f}' for c, p in zip(chars, probs))}")

    # Print tally
    print(f"\n{'=' * 80}")
    print("=== RESULTS ===")
    print(f"{'=' * 80}")
    for name, _, _ in models:
        t = tallies[name]
        passed = t["perfect"] + t["oversplit"]
        print(f"  {name}: {t['perfect']} perfect + {t['oversplit']} oversplit = {passed}/{total} passed ({t['wrong']} wrong)")
    print()


def main():
    print("Loading cedict vocab...")
    cedict_vocab = load_cedict_vocab(CEDICT_PATH)
    print(f"  {len(cedict_vocab)} words")

    models = []

    # Baseline
    print(f"Loading baseline model ({BASELINE_MODEL})...")
    tok, mod = load_model(BASELINE_MODEL, DEVICE)
    models.append(("Baseline", tok, mod))

    # Fine-tuned (skip if checkpoint doesn't exist)
    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.exists():
        print(f"Loading fine-tuned model ({CHECKPOINT_PATH})...")
        tok, mod = load_model(CHECKPOINT_PATH, DEVICE)
        models.append(("Finetuned", tok, mod))
    else:
        print(f"  Fine-tuned checkpoint not found at {CHECKPOINT_PATH}, skipping")

    print(f"\nDevice: {DEVICE}")
    print(f"Test cases: {len(VALIDATED_TEST_CASES)}")

    run_eval(models, cedict_vocab, VALIDATED_TEST_CASES, DEVICE)


if __name__ == "__main__":
    main()
