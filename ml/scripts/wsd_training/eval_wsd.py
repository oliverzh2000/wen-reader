#!/usr/bin/env python3
"""Evaluate WSD model on hand-picked hard disambiguation examples.

Usage:
    python scripts/wsd_training/eval_wsd.py                          # bi-encoder (default)
    python scripts/wsd_training/eval_wsd.py --cross-encoder-en       # cross-encoder with English senses
    python scripts/wsd_training/eval_wsd.py --cross-encoder-zh       # cross-encoder with Chinese senses
    python scripts/wsd_training/eval_wsd.py --all                    # compare all available models
"""

import json
import re
from math import comb
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR.parent.parent
CEDICT_PATH = ML_DIR.parent / "cedict" / "cedict_ts.u8"
CACHE_PATH = ML_DIR / "data" / "translation_cache.json"

BIENCODER_PATH = ML_DIR / "models" / "wsd_finetuned" / "final"
BIENCODER_V2_SMALL_PATH = ML_DIR / "models" / "wsd_finetuned_biencoder_gte_v2_small" / "final"
BIENCODER_V2_BASE_PATH = ML_DIR / "models" / "wsd_biencoder_gte_base" / "final"
BIENCODER_V2_DISTILLED_BASE_PATH = ML_DIR / "models" / "wsd_distilled_gte_v2_small" / "20260529_021557" / "final"
BIENCODER_V2_LARGE_PATH = ML_DIR / "models" / "wsd_finetuned_biencoder_gte_v2_large" / "final"
BIENCODER_BGE_BASE_PATH = ML_DIR / "models" / "wsd_finetuned_biencoder_bge_v2_base" / "final"
BIENCODER_BGE_LARGE_PATH = ML_DIR / "models" / "wsd_finetuned_biencoder_bge_v2_large" / "final"
CROSSENCODER_EN_PATH = ML_DIR / "models" / "wsd_finetuned_crossencoder_en_bge" / "final"
CROSSENCODER_ZH_PATH = ML_DIR / "models" / "wsd_finetuned_crossencoder_zh_hfl_lambda" / "final"
ENTRIES_PATH = ML_DIR / "data" / "entries_after_merging.json"
TRANSLATION_CACHE_PATH = ML_DIR / "data" / "translation_cache.json"

# name, path, kind, quantize_model, quantize_emb
model_configs = [
    ("Bi-encoder v1 (zh)", BIENCODER_PATH, "biencoder", False, None),
    ("Bi-encoder v2 small (zh)", BIENCODER_V2_SMALL_PATH, "biencoder", False, None),
    ("Bi-encoder v2 small distilled (zh)", BIENCODER_V2_DISTILLED_BASE_PATH, "biencoder", False, None),
    ("Bi-encoder v2 small distilled int8+fp16emb", BIENCODER_V2_DISTILLED_BASE_PATH, "biencoder", True, "fp16"),
    ("Bi-encoder v2 small distilled int8+int8emb", BIENCODER_V2_DISTILLED_BASE_PATH, "biencoder", True, "int8"),
    ("Bi-encoder v2 base (zh)", BIENCODER_V2_BASE_PATH, "biencoder", False, None),
    ("Bi-encoder v2 base fp16emb", BIENCODER_V2_BASE_PATH, "biencoder", False, "fp16"),
    ("Bi-encoder v2 base int8emb", BIENCODER_V2_BASE_PATH, "biencoder", False, "int8"),
    ("Bi-encoder v2 base fp16model", BIENCODER_V2_BASE_PATH, "biencoder", "fp16", None),
    ("Bi-encoder v2 base fp16model+int8emb", BIENCODER_V2_BASE_PATH, "biencoder", "fp16", "int8"),
    ("Bi-encoder v2 base int8+fp16emb", BIENCODER_V2_BASE_PATH, "biencoder", True, "fp16"),
    ("Bi-encoder v2 base int8+int8emb", BIENCODER_V2_BASE_PATH, "biencoder", True, "int8"),
    ("Bi-encoder v2 large (zh)", BIENCODER_V2_LARGE_PATH, "biencoder", False, None),
    ("Bi-encoder BGE base (zh)", BIENCODER_BGE_BASE_PATH, "biencoder", False, None),
    ("Bi-encoder BGE large (zh)", BIENCODER_BGE_LARGE_PATH, "biencoder", False, None),
    ("Cross-encoder (en)", CROSSENCODER_EN_PATH, "crossencoder_en", False, None),
    ("Cross-encoder (zh)", CROSSENCODER_ZH_PATH, "crossencoder_zh", False, None),
]


def quantize_biencoder_int8(model: SentenceTransformer) -> SentenceTransformer:
    """Apply dynamic int8 weight quantization to all Linear layers (CPU only)."""
    torch.backends.quantized.engine = "qnnpack"
    model = model.to("cpu")
    model[0].auto_model = torch.quantization.quantize_dynamic(
        model[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    return model


def quantize_biencoder_fp16(model: SentenceTransformer) -> SentenceTransformer:
    """Cast model to fp16 (simulates CoreML fp16 inference)."""
    model = model.half()
    model.eval()
    return model


def quantize_embeddings_fp16(embs: np.ndarray) -> np.ndarray:
    """Simulate fp16 storage: round-trip through float16."""
    return embs.astype(np.float16).astype(np.float32)


def quantize_embeddings_int8(embs: np.ndarray) -> np.ndarray:
    """Simulate int8 storage with per-vector scalar quantization.

    Each vector is scaled to [-127, 127], stored as int8, then dequantized.
    This matches what you'd do in the app: store scale + int8 vector per sense.
    """
    # Per-row scale: max absolute value
    scales = np.abs(embs).max(axis=1, keepdims=True)
    scales = np.where(scales == 0, 1.0, scales)  # avoid division by zero
    # Quantize
    quantized = np.round(embs / scales * 127.0).clip(-127, 127).astype(np.int8)
    # Dequantize
    return quantized.astype(np.float32) * scales / 127.0

# Hard examples: (word, context_with_markers, expected_english_substrings)
# expected is a set of acceptable English substrings — any match counts as correct.
#
# NOTE: Only includes words whose senses exist within a single CC-CEDICT entry's
# merged clusters. Polyphonic characters where different pronunciations live in
# separate entries (干 gān/gàn, 还 hái/huán, etc.) are excluded — the WSD model
# only disambiguates within an entry, not across entries.
EXAMPLES = [
    # === MODERN WORDS (same-entry polysemy) ===

    # 干 gàn: trunk vs do/work vs cadre vs kill vs fuck vs annoyed
    ("干", "这件事必须马上★干★，不能再拖了.", {"to do", "to work", "to manage"}),
    ("干", "他在工地上★干★了整整十年的苦力.", {"to work", "to do", "to manage"}),

    # 还 hái: still vs even vs also vs fairly vs as early as
    ("还", "这本书我★还★没看完，再借几天吧.", {"still", "yet"}),
    ("还", "他的成绩★还★算不错，能上重点大学.", {"fairly", "passably"}),

    # 发 fā: send/issue vs show feeling vs develop vs make money
    ("发", "地震★发★生的时候，所有人都跑出了大楼.", {"to send out", "to issue", "to develop"}),
    ("发", "他终于★发★了财，买了一栋别墅.", {"to make a bundle"}),

    # 得 dé: obtain vs proper vs proud vs allow vs ready
    ("得", "这次考试他★得★了满分.", {"to obtain", "to get", "to gain"}),

    # 好 hǎo: good vs all right vs easy to vs so vs completion vs close vs hello
    ("好", "这道菜做得真★好★，色香味俱全.", {"good"}),

    # 假 jiǎ: fake vs borrow vs if
    ("假", "这幅画是★假★的，不是真迹.", {"fake", "false", "artificial"}),

    # 薄 báo/bó: thin vs despise vs approach
    ("薄", "冬天穿这么★薄★的衣服会感冒的.", {"thin", "light"}),
    ("薄", "不可★薄★待远道而来的客人.", {"to despise", "to belittle", "to look down on", "ungenerous", "unkind"}),
    ("薄", "日★薄★西山，气息奄奄.", {"to approach", "to near"}),

    # 恶 wù: hate vs ashamed vs fear vs slander
    ("恶", "君子好善而★恶★不义.", {"to hate", "to loathe"}),

    # 鲜 xiān: fresh vs bright vs delicious vs fish
    ("鲜", "这条鱼非常★鲜★美，刚从海里捞上来的.", {"fresh", "delicious", "tasty"}),

    # 相 xiàng: appearance vs portrait vs minister vs phase
    ("相", "他官至宰★相★，权倾朝野.", {"minister"}),

    # 处 chù: place vs part vs office
    ("处", "此★处★风景绝佳，值得一游.", {"place", "area"}),

    # 落 luò: fall vs decline vs rest with vs get vs write down vs whereabouts vs settlement
    ("落", "秋风起，黄叶纷纷★落★下.", {"to fall", "to drop"}),

    # 藏 cáng: conceal vs store (multi-pinyin with zàng)
    ("藏", "他把秘密★藏★在心底，从未对人提起.", {"to conceal", "to hide", "to harbor"}),
    ("藏", "这位收藏家一生★藏★书数万卷.", {"to store", "to collect"}),

    # 弹 tán: pluck vs spring/flick vs accuse vs elastic
    ("弹", "她坐在窗前，轻轻地★弹★着古琴.", {"to pluck", "to play"}),

    # 尽 jǐn: to the greatest extent vs within limits vs give priority (3 clusters)
    ("尽", "请★尽★量多吃一点，别客气.", {"to the greatest extent", "furthest"}),
    ("尽", "★尽★着老人先用餐，我们年轻人不急.", {"to give priority to"}),

    # 间 jiān: between vs room (+ jiàn: gap vs sow discontent — multi-pinyin)
    ("间", "他们的关系被人从中挑拨离★间★.", {"to sow discontent", "to separate", "gap"}),
    ("间", "教室★间★里坐满了学生.", {"room", "section of a room"}),

    # 称 chēng: weigh vs state vs name vs praise
    ("称", "人们★称★他为当代最伟大的诗人.", {"to name", "to state"}),

    # 切 qiè: close/eager vs scoffing vs grind vs fanqie
    ("切", "老师的话语★切★中要害.", {"close to", "eager", "to correspond"}),

    # 兴 xīng: rise/flourish vs start vs encourage vs get up vs permit vs maybe
    ("兴", "国家★兴★亡，匹夫有责.", {"to rise", "to flourish"}),

    # 当 dàng: pawn vs regard as vs suitable vs replace vs fail
    ("当", "穷困潦倒之时，他只好去★当★铺典当衣物.", {"to pawn"}),

    # 便 biàn: plain vs convenient vs urinate vs then vs even if
    ("便", "交通★便★利是这个小区最大的优势.", {"convenient", "suitable"}),

    # === LITERARY / CLASSICAL WORDS ===

    # 负 — carry vs defeated vs betray
    ("负", "他★负★剑远游，踏遍天涯.", {"to carry", "to bear"}),
    ("负", "将军兵败，★负★于敌手.", {"to be defeated"}),
    ("负", "忘恩★负★义之人，天下共诛之.", {"to turn one's back on"}),

    # 绝 — cut off vs absolutely vs extinct
    ("绝", "悬崖★绝★壁之上，孤松傲立.", {"to cut short"}),
    ("绝", "此人武艺★绝★伦，天下无双.", {"absolutely", "by no means"}),
    ("绝", "这种鸟类已经★绝★迹多年.", {"extinct", "to disappear", "to vanish"}),

    # 穷 — poor vs exhaust
    ("穷", "他虽然★穷★，却从不向人借钱.", {"poor", "destitute"}),
    ("穷", "★穷★尽一切办法也要找到真相.", {"to exhaust", "to use up", "thoroughly"}),

    # 素 — plain vs usually vs vegetarian
    ("素", "她一身★素★衣，不施粉黛.", {"plain", "white"}),
    ("素", "他★素★来不问世事，隐居山林.", {"usually", "always", "ever"}),
    ("素", "这家餐厅只做★素★菜，不用荤腥.", {"vegetarian"}),

    # 度 dù: pass/spend vs measure/degree vs kilowatt-hour
    ("度", "他在寺庙中★度★过了清苦的一生.", {"to pass", "to spend"}),

    # 济 jì: aid vs cross a river (binary but classical distinction)
    ("济", "同舟共★济★，共渡难关.", {"to aid", "to relieve", "to be of help"}),

    # 遂 — succeed vs then
    ("遂", "他多年的心愿终于★遂★了.", {"to satisfy", "to succeed", "finally"}),
    ("遂", "事已至此，★遂★不再追究.", {"then", "thereupon", "to proceed"}),

    # 故 — therefore vs old
    ("故", "★故★君子慎其独也.", {"therefore"}),
    ("故", "温★故★而知新.", {"old"}),

    # 道 — road vs the Way vs to say
    ("道", "山间小★道★蜿蜒曲折.", {"road", "path"}),
    ("道", "天地有大★道★，万物循之而生.", {"the Way", "the Dao", "reason", "principle"}),
    ("道", "他微微一笑，★道★：'不必多言'.", {"to say"}),

    # 见 — to appear vs opinion
    ("见", "读书百遍，其义自★见★.", {"to appear"}),
    ("见", "依我之★见★，此事不可操之过急.", {"opinion", "view"}),

    # 望 — gaze vs hope
    ("望", "他站在山顶，★望★着远方的大海.", {"to gaze", "to look towards"}),
    ("望", "父母★望★子成龙，用心良苦.", {"to hope", "to expect"}),

    # 信 — letter vs trust vs at will
    ("信", "他写了一封长★信★，寄往故乡.", {"letter", "mail"}),
    ("信", "轻诺寡★信★，多易必多难.", {"confidence", "trust", "truthful"}),
    ("信", "★信★手拈来，皆成文章.", {"at random", "at will"}),

    # 经 — classics vs undergo
    ("经", "他熟读四书五★经★，学问渊博.", {"classics", "sacred book", "scripture"}),
    ("经", "★经★历了战火的洗礼，他变得沉默寡言.", {"to undergo", "to pass through", "to bear", "to endure"}),

    # 生 — life vs raw vs grow
    ("生", "他一★生★坎坷，却从未放弃.", {"life"}),
    ("生", "这块肉还是★生★的，不能吃.", {"raw", "uncooked"}),
    ("生", "春风吹过，万物★生★长.", {"to grow"}),

    # 明 — bright vs understand vs next
    ("明", "月色★明★亮，照得大地如同白昼.", {"bright"}),
    ("明", "他深★明★大义，不计个人得失.", {"to understand", "clear", "wise"}),
    ("明", "★明★年春天我们再相聚.", {"next"}),

    # 正 zhèng: straight/upright vs main vs correct vs exactly vs positive
    ("正", "为人★正★直，不阿谀奉承.", {"upright", "straight", "proper"}),

    # 为 wéi: to act as vs to be vs to become vs to make (4 clusters, multi-pinyin with wèi)
    ("为", "★为★天地立心，为生民立命.", {"for"}),
    ("为", "他年纪轻轻就★为★人师表.", {"to act as", "to be", "to become"}),

    # 乘 shèng: chariot vs four vs history books
    ("乘", "千★乘★之国，不可轻侮.", {"chariot"}),

    # === MULTI-CHARACTER AMBIGUOUS WORDS ===

    # 结果 — result vs kill
    ("结果", "★结果★出乎所有人的意料.", {"outcome", "result", "consequence"}),
    ("结果", "刺客趁夜潜入，将他★结果★了.", {"to kill", "to dispatch"}),

    # 精神 — spirit vs vigor
    ("精神", "这部作品体现了时代的★精神★.", {"spirit", "mind", "essence"}),
    ("精神", "老人虽年过八旬，依然★精神★矍铄.", {"vigor", "vitality", "spirited"}),

    # 风流 — distinguished vs dissolute
    ("风流", "数★风流★人物，还看今朝.", {"distinguished", "accomplished", "outstanding"}),
    ("风流", "他年轻时颇为★风流★，留下不少风流韵事.", {"dissolute", "loose", "romantic"}),

    # 老实 — honest vs naive (2 clusters, but tricky)
    ("老实", "他为人★老实★，从不说谎.", {"honest", "well-behaved"}),
    ("老实", "他太★老实★了，总是被人骗.", {"naive", "gullible"}),

    # 打发 — dispatch vs make leave vs pass time vs tip (4 clusters)
    ("打发", "主人★打发★仆人去集市买菜.", {"to dispatch"}),
    ("打发", "她随便找了个借口把他★打发★走了.", {"to make sb leave"}),
    ("打发", "下雨天无事可做，只好看书★打发★时间.", {"to pass", "to spend"}),

    # 方便 — convenient vs relieve oneself vs facilitate (3 clusters)
    ("方便", "网上购物非常★方便★，足不出户就能买到.", {"convenient", "suitable"}),
    ("方便", "他找了个隐蔽的角落★方便★了一下.", {"to relieve oneself"}),

    # 高低 — height vs superiority vs no matter what
    ("高低", "两座山峰★高低★错落，景色壮观.", {"height", "level", "pitch"}),
    ("高低", "两人棋艺相当，一时难分★高低★.", {"relative superiority"}),
    ("高低", "他★高低★不肯答应这门亲事.", {"no matter what", "simply"}),

    # 长短 — length vs mishap vs merits
    ("长短", "请量一下这块布的★长短★.", {"length"}),
    ("长短", "万一他有个★长短★，这个家就完了.", {"mishap", "accident"}),
    ("长短", "背后议论别人的★长短★，不是君子所为.", {"right and wrong", "good and bad", "merits"}),

    # 文章 — article vs hidden meaning
    ("文章", "他发表了一篇关于量子力学的★文章★.", {"article", "essay", "literary works"}),
    ("文章", "这件事背后大有★文章★，不可掉以轻心.", {"hidden meaning"}),

    # 手脚 — limbs vs trick vs step
    ("手脚", "天太冷了，★手脚★都冻僵了.", {"hands and feet", "movement", "action"}),
    ("手脚", "有人在账目上动了★手脚★.", {"trick"}),

    # 交代 — instruct vs confess
    ("交代", "临走前他★交代★了几件要紧的事.", {"to give instructions", "to tell", "to explain", "to transfer"}),
    ("交代", "在审讯室里，嫌疑人终于★交代★了罪行.", {"to confess", "to account for"}),

    # 打量 — size up vs suppose
    ("打量", "她上下★打量★了来人一番.", {"to size up", "to look sb over"}),
    ("打量", "你别★打量★我不知道你的心思.", {"to suppose", "to reckon"}),

    # 意思 — meaning vs token vs fun
    ("意思", "这句话是什么★意思★", {"meaning", "idea"}),
    ("意思", "这点小礼物不成敬意，只是个★意思★.", {"token of appreciation", "gesture of goodwill"}),
    ("意思", "这部电影挺有★意思★的.", {"fun", "interest"}),

    # === FUNCTION WORDS / PARTICLES ===

    # 就 jiu4: then vs merely vs approach vs undertake vs concerning vs go with
    ("就", "你要是不想去，★就★别去了。", {"(after a suppositional clause) in that case; then"}),
    ("就", "他一到家★就★开始做饭。", {"(after a clause of action) as soon as; immediately after"}),
    ("就", "这件事★就★他一个人知道。", {"merely", "only", "just"}),
    ("就", "★就★这个问题，我们讨论了很久。", {"with regard to", "concerning"}),
    ("就", "吃面条★就★着蒜瓣儿，特别香。", {"(of food) to go with"}),

    # 过 guo4: cross vs pass time vs celebrate vs live vs excessively
    ("过", "小心翼翼地★过★了那座独木桥。", {"to cross"}),
    ("过", "时间★过★得真快，转眼就到年底了。", {"to pass (time)"}),
    ("过", "今年我们回老家★过★春节。", {"to celebrate (a holiday)"}),
    ("过", "日子虽然清苦，但也能★过★得下去。", {"to live", "to get along"}),
    ("过", "这道菜盐放★过★了，太咸了。", {"excessively", "too-"}),

    # 给 gei3: to/for vs give vs allow vs passive marker
    ("给", "他★给★我写了一封信。", {"to", "for", "for the benefit of"}),
    ("给", "妈妈★给★了他一百块钱。", {"to give"}),
    ("给", "别★给★他知道这件事。", {"to allow"}),
    ("给", "花瓶★给★猫打碎了。", {"(passive or object marker)", "(intensifier)"}),

    # 被 bei4: quilt vs passive marker
    ("被", "天冷了，多盖一床★被★子。", {"quilt"}),
    ("被", "他★被★老师批评了一顿。", {"passive voice marker"}),

    # 把 ba3: hold vs classifier vs object marker
    ("把", "他紧紧★把★住栏杆，不敢往下看。", {"to hold", "to grasp"}),
    ("把", "给我来一★把★花生。", {"classifier: handful, bundle, bunch"}),
    ("把", "她★把★书放在桌子上。", {"(used to put the object before the verb"}),

    # 的 de5: possessive vs emphasis vs nominalizer
    ("的", "这是我★的★书，不是你的。", {"possessive", "of"}),
    ("的", "我是昨天来★的★。", {"(used at the end of a declarative sentence for emphasis)"}),

    # 了 liao3: finish/able vs clear-sighted
    ("了", "这件事没★了★，我们走着瞧。", {"to finish"}),
    ("了", "他一目★了★然地看穿了对方的把戏。", {"to understand clearly"}),

    # 了 le5: completed action vs change of state
    ("了", "他吃★了★三碗饭。", {"(completed action marker)"}),
    ("了", "天黑★了★，我们回家吧。", {"modal particle indicating change of state"}),

    # 着 zhao2: touch vs catch fire vs fall asleep vs succeed
    ("着", "别★着★凉了，多穿点衣服。", {"to feel", "to be affected by"}),
    ("着", "纸一碰到火就★着★了。", {"to catch fire", "to burn"}),
    ("着", "他太累了，一躺下就★着★了。", {"(coll.) to fall asleep"}),

    # 看 kan4: see/read vs visit vs consider vs depend on vs give it a try
    ("看", "我昨天★看★了一部很好的电影。", {"to see", "to look at", "to read", "to watch"}),
    ("看", "周末我去医院★看★望了生病的朋友。", {"to visit", "to call on"}),
    ("看", "我★看★这件事没那么简单。", {"to consider", "to feel (that)"}),
    ("看", "能不能成功，就★看★你自己的努力了。", {"to depend on"}),
    ("看", "你尝尝★看★，这道菜味道怎么样？", {"(after a verb) to give it a try"}),

    # 让 rang4: yield vs permit vs make sb do vs passive
    ("让", "互相★让★一步，事情就好办了。", {"to yield"}),
    ("让", "妈妈不★让★我出去玩。", {"to permit", "to let"}),
    ("让", "这部电影★让★我感动得流泪。", {"to have sb do sth", "to make sb feel"}),
    ("让", "杯子★让★他打碎了。", {"by (indicates the agent in a passive clause"}),

    # 会 hui4: can/know how vs likely vs meet vs association
    ("会", "她★会★弹钢琴，也会拉小提琴。", {"can", "to have the skill", "to know how to"}),
    ("会", "明天★会★下雨吗？", {"to be likely to", "to be sure to"}),
    ("会", "我们约好下周★会★面。", {"to meet", "meeting"}),

    # 要 yao4: want/need vs will/about to vs if vs important
    ("要", "我★要★一杯咖啡。", {"to want", "to need"}),
    ("要", "火车★要★开了，快上车！", {"will", "shall", "about to"}),
    ("要", "★要★是明天下雨，我们就不去了。", {"if"}),

    # 以 yi3: by means of vs according to vs in order to vs because of
    ("以", "★以★德报怨，何以报德？", {"to use", "by means of"}),
    ("以", "★以★我之见，此事不可为。", {"according to"}),
    ("以", "努力学习，★以★报效国家。", {"in order to"}),

    # 从 cong2: from vs follow vs engage in vs ever
    ("从", "他★从★北京坐火车到上海。", {"from", "through", "via"}),
    ("从", "★从★善如流，知错能改。", {"to follow", "to obey"}),
    ("从", "他★从★事教育工作已经二十年了。", {"to engage in (an activity)"}),
    ("从", "他★从★来没有迟到过。", {"(used before a negative) ever"}),

    # 而 er2: and vs therefore vs but
    ("而", "他聪明★而★勤奋，成绩一直名列前茅。", {"and", "as well as"}),
    ("而", "因贫穷★而★辍学。", {"and so", "therefore"}),
    ("而", "他嘴上答应了，★而★心里并不情愿。", {"but", "yet", "however"}),

    # 向 xiang4: towards vs support vs formerly vs always
    ("向", "他★向★东走去，消失在晨雾中。", {"towards", "to face", "direction"}),
    ("向", "在这场争论中，大多数人★向★着他。", {"to support", "to side with"}),
    ("向", "他★向★来不喜欢热闹的场合。", {"always", "all along"}),

    # 都 dou1: all vs even vs already
    ("都", "同学们★都★到齐了，可以开始上课了。", {"all", "both", "entirely"}),
    ("都", "连小孩子★都★知道这个道理。", {"(used for emphasis) even"}),
    ("都", "★都★什么时候了，你还在玩游戏！", {"already"}),

    # 再 zai4: again vs further vs then vs no matter how
    ("再", "这本书太好了，我想★再★看一遍。", {"again", "once more"}),
    ("再", "请★再★说详细一点。", {"further", "more"}),
    ("再", "等雨停了★再★走吧。", {"then (after sth, and not until then)"}),
    ("再", "★再★难也要坚持下去。", {"no matter how"}),

    # === ADDITIONAL COMMON POLYSEMOUS WORDS ===

    # 点 dian3: 13 clusters — tap, check off, order, mention, hint, drops, light, nod, dot, stroke, decimal, time, a bit
    ("点", "服务员，我要★点★菜。", {"to order", "to select"}),
    ("点", "请★点★一下人数，看看都到齐了没有。", {"to check off", "to mark with a dot"}),
    ("点", "他★点★了一支烟，沉默不语。", {"to light", "to ignite"}),
    ("点", "老师★点★了几个同学的名字。", {"to mention", "to bring up"}),
    ("点", "她微微★点★了点头，表示同意。", {"to nod"}),
    ("点", "今天有★点★冷，多穿一件外套吧。", {"a small amount", "a bit"}),

    # 白 bai2: 11 clusters — white, bright, empty, clear, in vain, reactionary, funeral, stare, wrong char, state, vernacular
    ("白", "她穿了一件★白★色的连衣裙。", {"white", "snowy", "pure"}),
    ("白", "我跟他说了半天，全★白★费了。", {"in vain", "gratuitous", "free of charge"}),
    ("白", "你把这件事跟他说★白★了没有？", {"clear", "to make clear"}),
    ("白", "这张纸还是★白★的，什么都没写。", {"empty", "blank", "plain"}),
    ("白", "他★白★了她一眼，没有搭话。", {"to stare coldly"}),

    # 行 xíng: 7 clusters — walk, temporary, current, do, capable, conduct, about to
    ("行", "他们沿着河边慢慢地★行★走。", {"to walk", "to travel", "trip"}),
    ("行", "你说的这个方案★行★不★行★？", {"capable", "all right"}),
    ("行", "此★行★目的已达，可以回去了。", {"to walk", "to travel", "trip"}),
    ("行", "大雨将★行★，乌云密布。", {"about to", "soon"}),

    # 行 háng: 4 clusters — row, trade, firm, rank
    ("行", "他在银★行★工作了二十年。", {"commercial firm"}),
    ("行", "入★行★三年，他已经是这个领域的专家了。", {"line of business", "trade", "profession"}),

    # 将 jiāng: 5 clusters — will, to use/take, checkmate, just now, object marker
    ("将", "我们★将★在明天出发。", {"will", "shall"}),
    ("将", "他★将★信递给了我。", {"to use", "to take"}),
    ("将", "★将★军！你的王无路可走了。", {"to checkmate"}),

    # 将 jiàng: general vs command vs chess piece
    ("将", "他是一位久经沙场的老★将★。", {"a general"}),

    # 对 dui4: 9 clusters — right, towards, treat, face, match, answer, add, check, pair
    ("对", "你说得★对★，我完全同意。", {"right", "correct"}),
    ("对", "他★对★陌生人总是很警惕。", {"towards", "regarding"}),
    ("对", "这双鞋跟你的衣服很★对★。", {"to match", "to suit"}),
    ("对", "记者★对★他提出了尖锐的问题。", {"to face", "opposite"}),
    ("对", "请把这两份文件★对★一下。", {"to check", "to compare"}),

    # 分 fen1: 8 clusters — divide, distinguish, branch, fraction, unit, minute, point, money
    ("分", "把这块蛋糕★分★成八份。", {"to divide", "to distribute"}),
    ("分", "你要学会★分★辨是非。", {"to distinguish"}),
    ("分", "比赛最后一★分★钟，他进了一球。", {"minute"}),
    ("分", "这次考试他得了九十★分★。", {"a point"}),

    # 去 qu4: 8 clusters — go, last, send, remove, apart, die, play, complement
    ("去", "我明天★去★北京出差。", {"to go", "to go to"}),
    ("去", "把皮★去★掉再切。", {"to remove", "to get rid of", "to reduce"}),
    ("去", "他★去★年刚结的婚。", {"last", "just passed"}),
    ("去", "老人昨晚★去★了。", {"to die"}),

    # 别 bie2: 7 clusters — leave, differentiate, turn away, other, don't, fasten, category
    ("别", "临★别★之际，他紧紧握住了我的手。", {"to leave", "to part"}),
    ("别", "★别★动！举起手来！", {"don't"}),
    ("别", "★别★人的事少管。", {"other", "another", "different"}),
    ("别", "她在胸前★别★了一枚胸针。", {"to fasten with a pin", "to stick in"}),

    # 头 tou2: 7 clusters — head, hair, top/end, stub, chief, side, first
    ("头", "他★头★疼得厉害，去看了医生。", {"head"}),
    ("头", "走到路的尽★头★，就能看到海了。", {"top", "end", "beginning or end"}),
    ("头", "他是我们部门的★头★。", {"chief", "boss"}),
    ("头", "★头★一次见面，他就给我留下了深刻印象。", {"first", "leading"}),

    # 开 kai1: 6 clusters — open, start, boil, write out, away, carat
    ("开", "请★开★门，快递到了。", {"to open"}),
    ("开", "他★开★了一家餐厅。", {"to start", "to turn on", "to operate"}),
    ("开", "水★开★了，可以泡茶了。", {"to boil"}),
    ("开", "医生给他★开★了一张处方。", {"to write out"}),
    ("开", "快走★开★！别挡路。", {"away", "off"}),

    # 面 mian4: 6 clusters — face, side/surface, flour, noodles, soft, spineless
    ("面", "她的★面★容十分清秀。", {"face"}),
    ("面", "这个问题需要从多个★面★来分析。", {"side", "surface", "aspect"}),
    ("面", "妈妈用★面★粉做了馒头。", {"flour"}),
    ("面", "中午吃碗★面★吧。", {"noodles"}),

    # 走 zou3: 6 clusters — walk, visit, leave, die, through, change
    ("走", "每天★走★路上班锻炼身体。", {"to walk", "to go", "to move"}),
    ("走", "这幅画放久了，颜色★走★了。", {"to change"}),
    ("走", "他不辞而别，悄悄★走★了。", {"to leave", "to go away"}),
    ("走", "过年我们去★走★亲戚。", {"to visit"}),

    # 回 hui2: 6 clusters — circle, return, answer, Hui, time, chapter
    ("回", "他从国外★回★来了。", {"to go back", "to return"}),
    ("回", "这个问题我没法★回★答你。", {"to answer"}),
    ("回", "这是我第三★回★来这里了。", {"time"}),

    # 长 cháng: long vs strong point (+ zhǎng: chief/elder/grow — multi-pinyin)
    ("长", "这条河很★长★，有几百公里。", {"long", "length"}),
    ("长", "每个人都有自己的★长★处。", {"strong point", "to be good at"}),

    # 长 zhǎng: chief vs elder vs grow
    ("长", "他是这个村的村★长★。", {"chief", "head"}),
    ("长", "孩子们一天天★长★大了。", {"to grow", "to develop", "to increase"}),

    # 老 lao3: 5 clusters — prefix, old, always, very, tough
    ("老", "★老★张，你最近忙什么呢？", {"prefix used before the surname"}),
    ("老", "这棵树已经很★老★了，有上百年的历史。", {"old", "experienced", "of long standing"}),
    ("老", "他★老★是迟到，让人很头疼。", {"always", "all the time"}),
    ("老", "这块牛肉煮★老★了，嚼不动。", {"tough"}),

    # 作 zuò: 6 clusters — do, write, pretend, regard as, be, feel
    ("作", "不要弄虚★作★假。", {"to pretend", "to feign"}),
    ("作", "这首诗是李白所★作★。", {"to write", "writings"}),
    ("作", "他的伤口又开始★作★痛了。", {"to feel"}),

    # 来 lai2: 6 clusters — come, ever since, next, in order to, approximately, possibility complement
    ("来", "他从很远的地方★来★看我。", {"to come", "hither"}),
    ("来", "多年★来★，他一直默默奉献。", {"ever since", "for the past"}),
    ("来", "大约有一百★来★人参加了活动。", {"approximately"}),
    ("来", "我们要努力学习，★来★报效祖国。", {"in order to"}),

    # 方 fang1: 9 clusters — square, power, upright, direction, side, place, method, prescription, just
    ("方", "这个房间大约有二十平★方★米。", {"square"}),
    ("方", "前★方★有一个十字路口。", {"direction"}),
    ("方", "双★方★都做出了让步。", {"side", "party"}),
    ("方", "这个★方★法非常有效。", {"method"}),
    ("方", "我★方★才到，你等很久了吗？", {"just", "only"}),
]



def load_merged_entries(path: Path) -> dict[str, list[dict]]:
    """Load entries_after_merging.json, index by word.
    
    Returns {word: [{"en": ..., "zh": ..., "senses": [...], "pinyin": ...}, ...]}
    where each item is a cluster from a polysemous entry.
    
    sense_zh is built by looking up each raw English sense in translation_cache.json
    and joining with "；". This matches exactly what the training data uses.
    """
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)

    with open(TRANSLATION_CACHE_PATH, encoding="utf-8") as f:
        translation_cache = json.load(f)

    by_word: dict[str, list[dict]] = {}
    missing_translations = 0
    total_senses = 0

    for e in entries:
        word = e["word"]
        pinyin = e["pinyin"]
        clusters = e.get("clusters", [])
        if len(clusters) < 2:
            continue  # skip monosemous
        for c in clusters:
            en = c.get("en", "") or "; ".join(c["senses"])
            # Build zh from translation cache (same as training data)
            zh_parts = []
            for sense_en in c["senses"]:
                total_senses += 1
                cache_key = f"{word}|{pinyin}|{sense_en}"
                zh = translation_cache.get(cache_key)
                if zh:
                    zh_parts.append(zh)
                else:
                    missing_translations += 1
            zh_combined = "；".join(zh_parts) if zh_parts else ""
            by_word.setdefault(word, []).append({
                "en": en,
                "zh": zh_combined,
                "senses": c["senses"],
                "pinyin": pinyin,
            })

    if missing_translations:
        print(f"  WARNING: {missing_translations}/{total_senses} senses missing from translation_cache")

    return by_word


def get_merged_senses(word: str, merged: dict) -> list[tuple[str, str, str]]:
    """Get merged cluster senses for a word. Returns [(english, chinese, pinyin), ...]"""
    clusters = merged.get(word, [])
    return [(c["en"], c["zh"], c["pinyin"]) for c in clusters]


def rank_biencoder(context, senses, model):
    """Rank using bi-encoder: encode context and Chinese cluster labels separately, dot product."""
    chinese_senses = [s[1] for s in senses]  # zh labels
    context_emb = model.encode(context, normalize_embeddings=True)
    sense_embs = model.encode(chinese_senses, normalize_embeddings=True)
    scores = context_emb @ sense_embs.T
    return scores


def rank_crossencoder(context, senses, model, sense_type):
    """Rank using cross-encoder: score (context, sense) pairs directly."""
    if sense_type == "en":
        sense_texts = [s[0] for s in senses]  # English
    else:
        sense_texts = [s[1] for s in senses]  # Chinese
    pairs = [(context, st) for st in sense_texts]
    scores = model.predict(pairs)
    return np.array(scores)


def batch_score_biencoder(model, merged, examples, emb_quantize=None):
    """Batch-score all examples for a bi-encoder. Returns list of score arrays (None if no senses).

    emb_quantize: None (fp32), "fp16", or "int8" — applies to sense embeddings only
                  (simulates precomputed sense storage at lower precision).
    """
    contexts = []
    valid_indices = []

    for i, (word, context, _) in enumerate(examples):
        senses = get_merged_senses(word, merged)
        if senses:
            contexts.append(context)
            valid_indices.append(i)

    # Batch encode all contexts at once
    context_embs = model.encode(contexts, normalize_embeddings=True, batch_size=64)

    # Cache sense embeddings per word (many examples share the same word)
    sense_emb_cache: dict[str, np.ndarray] = {}
    for ex_idx in valid_indices:
        word = examples[ex_idx][0]
        if word not in sense_emb_cache:
            senses = get_merged_senses(word, merged)
            chinese_senses = [s[1] for s in senses]
            embs = model.encode(chinese_senses, normalize_embeddings=True, batch_size=64)
            # Apply embedding quantization (simulates on-device storage)
            if emb_quantize == "fp16":
                embs = quantize_embeddings_fp16(embs)
            elif emb_quantize == "int8":
                embs = quantize_embeddings_int8(embs)
            sense_emb_cache[word] = embs

    # Compute scores via dot product
    all_scores = [None] * len(examples)
    for i, ex_idx in enumerate(valid_indices):
        word = examples[ex_idx][0]
        all_scores[ex_idx] = context_embs[i] @ sense_emb_cache[word].T

    return all_scores


def batch_score_crossencoder(model, sense_type, merged, examples):
    """Batch-score all examples for a cross-encoder. Returns list of score arrays (None if no senses)."""
    all_pairs = []
    pair_boundaries = []  # (example_idx, start, end)

    for i, (word, context, _) in enumerate(examples):
        senses = get_merged_senses(word, merged)
        if not senses:
            continue
        if sense_type == "en":
            sense_texts = [s[0] for s in senses]
        else:
            sense_texts = [s[1] for s in senses]
        start = len(all_pairs)
        all_pairs.extend((context, st) for st in sense_texts)
        pair_boundaries.append((i, start, len(all_pairs)))

    # Single batch predict
    all_predictions = model.predict(all_pairs, batch_size=64)

    # Scatter back
    all_scores = [None] * len(examples)
    for ex_idx, start, end in pair_boundaries:
        all_scores[ex_idx] = np.array(all_predictions[start:end])

    return all_scores


def _matches(expected, english):
    """Check if any expected substring matches the english sense."""
    if isinstance(expected, str):
        expected = {expected}
    return any(exp.lower() in english.lower() for exp in expected)


def run_eval(model_name, all_scores, merged, examples):
    """Run evaluation using pre-computed scores. Returns (correct, top3_correct, mrr_sum, total)."""
    correct = 0
    top3_correct = 0
    mrr_sum = 0.0
    total = 0

    for i, (word, context, expected) in enumerate(examples):
        senses = get_merged_senses(word, merged)
        if not senses:
            print(f"\n⚠️  No senses found for '{word}' - skipping")
            continue

        scores = all_scores[i]
        if scores is None:
            print(f"\n⚠️  No scores for '{word}' - skipping")
            continue

        ranked_indices = scores.argsort()[::-1]

        top1_english = senses[ranked_indices[0]][0]
        is_correct = _matches(expected, top1_english)

        top3_english = [senses[ranked_indices[j]][0] for j in range(min(3, len(ranked_indices)))]
        is_top3 = any(_matches(expected, eng) for eng in top3_english)

        if is_correct:
            correct += 1
        if is_top3:
            top3_correct += 1

        for rank, idx in enumerate(ranked_indices, 1):
            if _matches(expected, senses[idx][0]):
                mrr_sum += 1.0 / rank
                break

        total += 1

        expected_str = expected if isinstance(expected, str) else " | ".join(sorted(expected))
        status = "✓" if is_correct else ("◐" if is_top3 else "✗")
        print(f"\n{status} [{word}] {context}")
        print(f"   Expected: '{expected_str}'")
        print(f"   Ranked senses:")
        for rank, idx in enumerate(ranked_indices[:5], 1):
            eng, chn, pin = senses[idx]
            score = scores[idx]
            marker = " ← TOP" if rank == 1 else ""
            print(f"   {rank}. {score:.4f} | {eng} | {chn}{marker}")
        if len(senses) > 5:
            print(f"   ... ({len(senses) - 5} more senses)")

    return correct, top3_correct, mrr_sum, total


def main():
    print("Loading merged entries...")
    merged = load_merged_entries(ENTRIES_PATH)
    print(f"  {len(merged)} polysemous words loaded")

    models = []

    for name, path, kind, quantize_model, quantize_emb in model_configs:
        if not path.exists() or not (path / "config.json").exists():
            print(f"⚠️  {name} not found at {path}")
            continue
        print(f"Loading {name} from {path}...")
        if kind == "biencoder":
            m = SentenceTransformer(str(path))
            if quantize_model == True:
                m = quantize_biencoder_int8(m)
                print(f"  → Model weights quantized to int8")
            elif quantize_model == "fp16":
                m = quantize_biencoder_fp16(m)
                print(f"  → Model weights cast to fp16")
            models.append((name, "biencoder", m, quantize_emb))
        else:
            sense_type = "en" if kind == "crossencoder_en" else "zh"
            m = CrossEncoder(str(path))
            models.append((name, "crossencoder", m, sense_type))

    if not models:
        print("No models found.")
        return

    results = {}
    for name, kind, model, extra in models:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {name}")
        print(f"{'=' * 80}")

        # Batch score all examples at once
        if kind == "biencoder":
            all_scores = batch_score_biencoder(model, merged, EXAMPLES, emb_quantize=extra)
        else:
            all_scores = batch_score_crossencoder(model, extra, merged, EXAMPLES)

        results[name] = run_eval(name, all_scores, merged, EXAMPLES)

    # Random baseline (silent — only shows up in summary)
    print("\nComputing random baseline...")
    rng = np.random.default_rng(42)
    random_correct = 0
    random_top3 = 0
    random_mrr = 0.0
    random_total = 0
    n_trials = 1000  # average over many trials for stable estimate

    for word, context, expected in EXAMPLES:
        senses = get_merged_senses(word, merged)
        if not senses:
            continue
        n_senses = len(senses)
        # Find which indices are correct
        correct_indices = [i for i, s in enumerate(senses) if _matches(expected, s[0])]
        if not correct_indices:
            random_total += 1
            continue

        # Analytically compute expected metrics for random ordering
        # P(top-1 correct) = |correct| / n_senses
        p_top1 = len(correct_indices) / n_senses
        # P(top-3 correct) = 1 - P(none in top 3)
        # P(none in top 3) = C(n-|c|, min(3,n)) / C(n, min(3,n))
        k = min(3, n_senses)
        n_wrong = n_senses - len(correct_indices)
        if n_wrong < k:
            p_top3 = 1.0
        else:
            p_top3 = 1.0 - comb(n_wrong, k) / comb(n_senses, k)
        # E[MRR] = sum over rank r of P(first correct at rank r) * 1/r
        # Simpler: E[1/rank] for uniform random = (|c|/n) * sum_{r=1}^{n} (1/r) * P(first correct at r)
        # Just simulate it for simplicity
        mrr_trials = []
        for _ in range(n_trials):
            perm = rng.permutation(n_senses)
            for rank, idx in enumerate(perm, 1):
                if idx in correct_indices:
                    mrr_trials.append(1.0 / rank)
                    break

        random_correct += p_top1
        random_top3 += p_top3
        random_mrr += np.mean(mrr_trials)
        random_total += 1

    results["Random (baseline)"] = (random_correct, random_top3, random_mrr, random_total)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for name, (correct, top3, mrr, total) in results.items():
        print(f"  {name}: TOP-1={correct}/{total} ({100*correct/total:.1f}%)  TOP-3={top3}/{total} ({100*top3/total:.1f}%)  MRR={mrr/total:.3f}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
