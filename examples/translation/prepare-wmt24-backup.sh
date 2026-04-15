#!/usr/bin/env bash
set -euo pipefail

# ========= 0) 基本依赖 =========
[ -d mosesdecoder ] || git clone https://github.com/moses-smt/mosesdecoder.git
[ -d subword-nmt ]  || git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

# ========= 1) 可配置参数（按需覆盖为你的路径） =========
# 语料位置（你的结构：Statmt-europarl-10-deu-eng/train-parts/*.deu|*.eng）
CORPUS_DIR=${CORPUS_DIR:-orig/wmt24-eng-deu}
PARTS_DIR=${PARTS_DIR:-$CORPUS_DIR/train-parts}

# 语言后缀（支持 deu/eng 或 de/en）
SRC=${SRC:-deu}
TGT=${TGT:-eng}
# Moses 的语言码（前两位）
SRC2=${SRC2:-${SRC:0:2}}
TGT2=${TGT2:-${TGT:0:2}}

# 输出目录
OUT_ROOT=${OUT_ROOT:-europarl_deu_eng-tok}
TMP=$OUT_ROOT/tmp
PREP=$OUT_ROOT/prep
mkdir -p "$TMP" "$PREP"

# BPE merges 数
BPE_TOKENS=${BPE_TOKENS:-32000}
# 是否小写化（一般 WMT 评测保留大小写：false）
DO_LOWERCASE=${DO_LOWERCASE:-false}

# 当没有单独的 dev 时，从训练里抽样切一份 dev
MAKE_HOLDOUT_DEV=${MAKE_HOLDOUT_DEV:-true}
# 抽样规则：每隔多少行取一行进 dev（越小 dev 越大）
HOLDOUT_EVERY=${HOLDOUT_EVERY:-50}
# dev 最大行数（0 表示不限制）
HOLDOUT_MAX=${HOLDOUT_MAX:-4000}

# 可选：若你已有 dev 文件，直接指定它们（将跳过 holdout）
DEV_SRC=${DEV_SRC:-}   # e.g. /path/dev.$SRC
DEV_TGT=${DEV_TGT:-}

# ========= 2) 汇总 train-parts =========
echo ">>> concat train parts from: $PARTS_DIR"
zcat -f $(find "$PARTS_DIR" -type f -name "*.$SRC*" | sort) > "$TMP/train.$SRC" || true
zcat -f $(find "$PARTS_DIR" -type f -name "*.$TGT*" | sort) > "$TMP/train.$TGT" || true

if ! [ -s "$TMP/train.$SRC" ] || ! [ -s "$TMP/train.$TGT" ]; then
  echo "FATAL: train files not found or empty under $PARTS_DIR"
  exit 1
fi

# 行数一致性检查
n_src=$(wc -l < "$TMP/train.$SRC"); n_tgt=$(wc -l < "$TMP/train.$TGT")
if [ "$n_src" -ne "$n_tgt" ]; then
  echo "FATAL: line count mismatch: src=$n_src tgt=$n_tgt"
  exit 1
fi
echo ">>> raw train lines: $n_src"

# ========= 3) 分词（moses，可选小写） =========
echo ">>> tokenizing train..."
if [ "$DO_LOWERCASE" = true ]; then
  perl $TOKENIZER -threads 8 -l $SRC2 < "$TMP/train.$SRC" | perl $LC > "$TMP/train.tok.$SRC"
  perl $TOKENIZER -threads 8 -l $TGT2 < "$TMP/train.$TGT" | perl $LC > "$TMP/train.tok.$TGT"
else
  perl $TOKENIZER -threads 8 -l $SRC2 < "$TMP/train.$SRC" > "$TMP/train.tok.$SRC"
  perl $TOKENIZER -threads 8 -l $TGT2 < "$TMP/train.$TGT" > "$TMP/train.tok.$TGT"
fi

# ========= 4) 清洗（长度/比例；仅 train） =========
# 最小1，最大250，长度比 3.0（按需改）
echo ">>> cleaning train (length & ratio)..."
perl $CLEAN -ratio 3.0 "$TMP/train.tok" $SRC $TGT "$TMP/train.clean" 1 250

echo ">>> cleaned lines:"
wc -l "$TMP/train.clean.$SRC" "$TMP/train.clean.$TGT"

# ========= 5) 若无 dev，则切分一份 holdout dev =========
if [ -n "$DEV_SRC" ] && [ -n "$DEV_TGT" ]; then
  echo ">>> using provided dev files: $DEV_SRC , $DEV_TGT"
  cp -f "$DEV_SRC" "$TMP/dev.tok.$SRC"
  cp -f "$DEV_TGT" "$TMP/dev.tok.$TGT"
else
  if [ "$MAKE_HOLDOUT_DEV" = true ]; then
    echo ">>> making holdout dev: every $HOLDOUT_EVERY lines (max $HOLDOUT_MAX)"
    paste "$TMP/train.clean.$SRC" "$TMP/train.clean.$TGT" | \
      awk -v k="$HOLDOUT_EVERY" -v max="$HOLDOUT_MAX" 'BEGIN{c=0}
        { if (NR % k == 0 && (max==0 || c<max)) { print > "dev.tsv"; c++ } else { print > "train.tsv" } }'
    cut -f1 dev.tsv > "$TMP/dev.tok.$SRC";  cut -f2 dev.tsv > "$TMP/dev.tok.$TGT"
    cut -f1 train.tsv > "$TMP/train.clean2.$SRC"; cut -f2 train.tsv > "$TMP/train.clean2.$TGT"
    mv -f "$TMP/train.clean2.$SRC" "$TMP/train.clean.$SRC"
    mv -f "$TMP/train.clean2.$TGT" "$TMP/train.clean.$TGT"
    rm -f dev.tsv train.tsv
  else
    echo ">>> no dev will be created; training will use all data"
    : > "$TMP/dev.tok.$SRC"; : > "$TMP/dev.tok.$TGT"
  fi
fi

# ========= 6) 训练 BPE（共享；各语种各学一个 code） =========

# 不共享：
# echo ">>> learning separate BPE codes: $SRC ($BPE_TOKENS) and $TGT ($BPE_TOKENS)"
# BPE_CODE_SRC=$PREP/bpe.$SRC.$BPE_TOKENS
# BPE_CODE_TGT=$PREP/bpe.$TGT.$BPE_TOKENS

# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < "$TMP/train.clean.$SRC" > "$BPE_CODE_SRC"
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < "$TMP/train.clean.$TGT" > "$BPE_CODE_TGT"

# ========= 6) 训练 BPE（联合词表） =========
echo ">>> learning joint BPE with $BPE_TOKENS merges..."
cat "$TMP/train.clean.$SRC" "$TMP/train.clean.$TGT" > "$TMP/train.concat"
BPE_CODE=$PREP/bpe.$BPE_TOKENS
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < "$TMP/train.concat" > "$BPE_CODE"




# ========= 7) 应用 BPE（train/dev；各用各的 code） =========
# echo ">>> applying BPE (separate codes) ..."
# python $BPEROOT/apply_bpe.py -c "$BPE_CODE_SRC" < "$TMP/train.clean.$SRC" > "$PREP/train.$SRC"
# python $BPEROOT/apply_bpe.py -c "$BPE_CODE_TGT" < "$TMP/train.clean.$TGT" > "$PREP/train.$TGT"

# if [ -s "$TMP/dev.tok.$SRC" ] && [ -s "$TMP/dev.tok.$TGT" ]; then
#   python $BPEROOT/apply_bpe.py -c "$BPE_CODE_SRC" < "$TMP/dev.tok.$SRC" > "$PREP/valid.$SRC"
#   python $BPEROOT/apply_bpe.py -c "$BPE_CODE_TGT" < "$TMP/dev.tok.$TGT" > "$PREP/valid.$TGT"
# fi

# ========= 7) 应用 BPE 共享词表（train/dev；无 test 则跳过） =========
echo ">>> applying BPE..."
python $BPEROOT/apply_bpe.py -c "$BPE_CODE" < "$TMP/train.clean.$SRC" > "$PREP/train.$SRC"
python $BPEROOT/apply_bpe.py -c "$BPE_CODE" < "$TMP/train.clean.$TGT" > "$PREP/train.$TGT"

if [ -s "$TMP/dev.tok.$SRC" ] && [ -s "$TMP/dev.tok.$TGT" ]; then
  python $BPEROOT/apply_bpe.py -c "$BPE_CODE" < "$TMP/dev.tok.$SRC" > "$PREP/valid.$SRC"
  python $BPEROOT/apply_bpe.py -c "$BPE_CODE" < "$TMP/dev.tok.$TGT" > "$PREP/valid.$TGT"
fi


# 若将来加入公开 test（如 newstest2023），同理各自用各自的 code：
# python $BPEROOT/apply_bpe.py -c "$BPE_CODE_SRC" < test.$SRC > $PREP/test.$SRC
# python $BPEROOT/apply_bpe.py -c "$BPE_CODE_TGT" < test.$TGT > $PREP/test.$TGT


# 如果你稍后会接入公开 test（例如 newstest2023），可在外部另行准备并套用同一 BPE：
#   python $BPEROOT/apply_bpe.py -c $BPE_CODE < test.$SRC > $PREP/test.$SRC
#   python $BPEROOT/apply_bpe.py -c $BPE_CODE < test.$TGT > $PREP/test.$TGT

echo ">>> DONE. Outputs at: $PREP"
ls -lh "$PREP"


# 例：你的目录结构
#   /path/to/fairseq-upload/data/Statmt-europarl-10-deu-eng/train-parts/*.deu|*.eng



# 下载：
# mtdata get   -l deu-eng   -tr Statmt-europarl-10-deu-eng   --no-merge --compress   -o orig/wmt24-eng-deu/



# bash prepare-wmt24.sh \
#   CORPUS_DIR=/path/to/fairseq-upload/data/Statmt-europarl-10-deu-eng \
#   OUT_ROOT=/path/to/fairseq-upload/prep_europarl_deu_eng \
#   SRC=deu TGT=eng \
#   BPE_TOKENS=32000 \
#   DO_LOWERCASE=false \
#   MAKE_HOLDOUT_DEV=true HOLDOUT_EVERY=50 HOLDOUT_MAX=4000


# OUT_ROOT=${OUT_ROOT:-europarl_deu_eng-tok}
# PREP=$OUT_ROOT/prep
# fairseq-preprocess \
#   --source-lang deu --target-lang eng \
#   --trainpref $PREP/train \
#   --validpref $PREP/valid \
#   --destdir ../../data-bin/europarl.deu-eng.bpe-separate \
#   --workers 16 \
#   --joined-dictionary