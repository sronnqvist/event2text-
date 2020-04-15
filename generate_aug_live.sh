TMP=$1
INPUT=$2
OUTPUT=$3
GPU=$4
SESSION=$(echo $INPUT|cut -d"." -f1)
outmax=100
BEST=../models/gen_model_v2.pt
SP_MODEL=../models/sentpiece.model
BS=128
ALPHA=1.6

SESSION_DIR=$(echo $TMP/tmp.$SESSION)

generate(){
  python OpenNMT-py/translate.py -seed $3 -gpu $GPU -model $BEST -src $TMP/$INPUT.pcs -output $SESSION_DIR/test_pred_$2.$3.txt -replace_unk -max_length $outmax -batch_size $BS -min_length $2 -length_penalty wu -alpha $ALPHA -verbose -attn_debug -beam_size 15 > $SESSION_DIR/trans.output.$2.$3
  python calc_attn_errors.py -i $SESSION_DIR/trans.output.$2.$3 -o $SESSION_DIR/scoring_$2.$3.txt
  grep -n "" $SESSION_DIR/scoring_$2.$3.txt |sed -E "s/(^[0-9]+):/\1\t/g" |while read LINE;do echo -e "$1\t$2\t$LINE";done > $SESSION_DIR/scoring_$2.$3.txt_
  mv $SESSION_DIR/scoring_$2.$3.txt_ $SESSION_DIR/scoring_$2.$3.txt
}

python event-augment/data2pieces_single.py $TMP/$INPUT "" $SP_MODEL

mkdir $SESSION_DIR
#rm tmp/*
# Manual test original entities with varying seed
echo "Generating text based on $TMP/$INPUT"
for i in 1 2 3 4 5 6; do
  length=$(echo "long long medium medium short short"|cut -d" " -f$i)
  min_length=$(echo "25 20 15 10 7 0"|cut -d" " -f$i)
  for s in 1; do #2 3
    seed=$(echo "10101 20202 30303"|cut -d" " -f$s)
    generate $length $min_length $seed &
  done
done
wait
# Join candidates and select; detokenize
echo "Writing final generation to $TMP/$OUTPUT"
paste -d "\n" $SESSION_DIR/scoring_*.*.txt |python rank_generation.py |sed "s/ //g" |sed "s/▁/ /g" |sed "s/ – /–/g" |sed "s/ - /-/g" |sed "s/ — /—/g" |sed "s/ : /:/g" |sed "s/ \./\./g" |sed "s/ ,/,/g" |sed "s/( /(/g" |sed "s/ )/)/g" > $TMP/$OUTPUT
rm -rf $SESSION_DIR
