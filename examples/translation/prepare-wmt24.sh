#!/usr/bin/env bash
# ========= 0) 基本依赖 =========
# [ -d mosesdecoder ] || git clone https://github.com/moses-smt/mosesdecoder.git
# [ -d subword-nmt ]  || git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

# ========= 1) 可配置参数 =========
CORPUS_DIR=${CORPUS_DIR:-/path/to/fairseq-upload/examples/translation/orig/wmt-news-19}

SRC=${SRC:-eng}
TGT=${TGT:-deu}
SRC2=${SRC2:-${SRC:0:2}}
TGT2=${TGT2:-${TGT:0:2}}

OUT_ROOT=${OUT_ROOT:-wmt19_eng_deu_tok}
TMP=$OUT_ROOT/tmp
PREP=$OUT_ROOT/prep
mkdir -p "$TMP" "$PREP"

BPE_TOKENS=${BPE_TOKENS:-32000}
DO_LOWERCASE=${DO_LOWERCASE:-false}

MAKE_HOLDOUT_DEV=${MAKE_HOLDOUT_DEV:-true}
HOLDOUT_EVERY=${HOLDOUT_EVERY:-50}
HOLDOUT_MAX=${HOLDOUT_MAX:-4000}

DEV_SRC=${DEV_SRC:-}
DEV_TGT=${DEV_TGT:-}

# ========= 2) 定位原始文件 =========
TRAIN_SRC=$CORPUS_DIR/train.$SRC
TRAIN_TGT=$CORPUS_DIR/train.$TGT

if ! [ -s "$TRAIN_SRC" ] || ! [ -s "$TRAIN_TGT" ]; then
  echo "FATAL: $TRAIN_SRC or $TRAIN_TGT not found"
  exit 1
fi

n_src=$(wc -l < "$TRAIN_SRC"); n_tgt=$(wc -l < "$TRAIN_TGT")
if [ "$n_src" -ne "$n_tgt" ]; then
  echo "FATAL: line mismatch: src=$n_src tgt=$n_tgt"
  exit 1
fi
echo ">>> raw train lines: $n_src"

# ========= 3) 分词 =========
echo ">>> tokenizing train..."
if [ "$DO_LOWERCASE" = true ]; then
  perl $TOKENIZER -threads 8 -l $SRC2 < "$TRAIN_SRC" | perl $LC > "$TMP/train.tok.$SRC"
  perl $TOKENIZER -threads 8 -l $TGT2 < "$TRAIN_TGT" | perl $LC > "$TMP/train.tok.$TGT"
else
  perl $TOKENIZER -threads 8 -l $SRC2 < "$TRAIN_SRC" > "$TMP/train.tok.$SRC"
  perl $TOKENIZER -threads 8 -l $TGT2 < "$TRAIN_TGT" > "$TMP/train.tok.$TGT"
fi

# ========= 4) 清洗 =========
echo ">>> cleaning train..."
perl $CLEAN -ratio 3.0 "$TMP/train.tok" $SRC $TGT "$TMP/train.clean" 1 250

# ========= 5) 划分 test (10%) 和 remainder (90%) =========
echo ">>> split test 10%..."
paste "$TMP/train.clean.$SRC" "$TMP/train.clean.$TGT" | \
  awk '{ if (NR % 10 == 0) print > "test.tsv"; else print > "remainder.tsv" }'

cut -f1 test.tsv > "$TMP/test.tok.$SRC";  cut -f2 test.tsv > "$TMP/test.tok.$TGT"
cut -f1 remainder.tsv > "$TMP/remainder.tok.$SRC"; cut -f2 remainder.tsv > "$TMP/remainder.tok.$TGT"
rm -f test.tsv remainder.tsv

# ========= 6) 从 remainder 切 dev =========
if [ -n "$DEV_SRC" ] && [ -n "$DEV_TGT" ]; then
  cp -f "$DEV_SRC" "$TMP/dev.tok.$SRC"
  cp -f "$DEV_TGT" "$TMP/dev.tok.$TGT"
  mv -f "$TMP/remainder.tok.$SRC" "$TMP/train.final.$SRC"
  mv -f "$TMP/remainder.tok.$TGT" "$TMP/train.final.$TGT"
else
  if [ "$MAKE_HOLDOUT_DEV" = true ]; then
    paste "$TMP/remainder.tok.$SRC" "$TMP/remainder.tok.$TGT" | \
      awk -v k="$HOLDOUT_EVERY" -v max="$HOLDOUT_MAX" 'BEGIN{c=0}
        { if (NR % k == 0 && (max==0 || c<max)) { print > "dev.tsv"; c++ } else { print > "train.tsv" } }'
    cut -f1 dev.tsv > "$TMP/dev.tok.$SRC"; cut -f2 dev.tsv > "$TMP/dev.tok.$TGT"
    cut -f1 train.tsv > "$TMP/train.final.$SRC"; cut -f2 train.tsv > "$TMP/train.final.$TGT"
    rm -f dev.tsv train.tsv
  else
    mv -f "$TMP/remainder.tok.$SRC" "$TMP/train.final.$SRC"
    mv -f "$TMP/remainder.tok.$TGT" "$TMP/train.final.$TGT"
    : > "$TMP/dev.tok.$SRC"; : > "$TMP/dev.tok.$TGT"
  fi
fi

# ========= 7) 学习联合 BPE (只用训练集) =========
echo ">>> learning joint BPE..."
cat "$TMP/train.final.$SRC" "$TMP/train.final.$TGT" > "$TMP/train.final.concat"
BPE_CODE=$PREP/bpe.$BPE_TOKENS
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < "$TMP/train.final.concat" > "$BPE_CODE"

# ========= 8) 应用 BPE 到 train/dev/test =========
# ========= 8) 应用 BPE 到 train/dev/test =========
echo ">>> applying BPE to train/dev/test..."
for SPLIT in train.final dev test; do
  for LANG in $SRC $TGT; do
    case "$SPLIT" in
      train.final) IN_FILE=$TMP/train.final.$LANG ; OUT_FILE=$PREP/train.$LANG ;;
      dev)         IN_FILE=$TMP/dev.tok.$LANG     ; OUT_FILE=$PREP/valid.$LANG ;;
      test)        IN_FILE=$TMP/test.tok.$LANG    ; OUT_FILE=$PREP/test.$LANG  ;;
    esac
    [ -s "$IN_FILE" ] || continue
    python $BPEROOT/apply_bpe.py -c "$BPE_CODE" < "$IN_FILE" > "$OUT_FILE"
  done
done



bash prepare-wmt24.sh \
  CORPUS_DIR=/path/to/fairseq-upload/examples/translation/orig/wmt-news-19 \
  OUT_ROOT=/path/to/fairseq-upload/wmt-news-19 \
  SRC=deu TGT=eng \
  BPE_TOKENS=32000 \
  DO_LOWERCASE=false \
  MAKE_HOLDOUT_DEV=true HOLDOUT_EVERY=50 HOLDOUT_MAX=4000


OUT_ROOT=${OUT_ROOT:-wmt19_eng_deu_tok}
PREP=$OUT_ROOT/prep
fairseq-preprocess \
  --source-lang eng --target-lang deu \
  --trainpref $PREP/train \
  --validpref $PREP/valid \
  --testpref $PREP/test \
  --destdir ../../data-bin/wmt19_eng_deu_tok \
  --workers 16 \
  --joined-dictionary

