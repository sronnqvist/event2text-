outmax=100
BEST=trained_model.pt
BS=128
ALPHA=1.6

generate(){
  python OpenNMT-py/translate.py -seed $3 -gpu 0 -model $BEST -src data/test_manual_$1.input.pcs -output tmp/test_pred_$2.$3.txt -replace_unk -max_length $outmax -batch_size $BS -min_length $2 -length_penalty wu -alpha $ALPHA -verbose -attn_debug -beam_size 15 > tmp/trans.output.$2.$3
  #grep "PRED SCORE" trans.output | cut -d":" -f2 > tmp/test_scores_$length.$s.txt
  python calc_attn_errors.py -i tmp/trans.output.$2.$3 -o tmp/scoring_$2.$3.txt
  grep -n "" tmp/scoring_$2.$3.txt |sed -E "s/(^[0-9]+):/\1\t/g" |while read LINE;do echo -e "$1\t$2\t$LINE";done > tmp/scoring_$2.$3.txt_
  #cat tmp/scoring_$min_length.$s.txt |while read LINE;do echo -e "$length\t$min_length\t$LINE";done > tmp/scoring_$min_length.$s.txt_
  mv tmp/scoring_$2.$3.txt_ tmp/scoring_$2.$3.txt
  #head tmp/scoring_$2.$3.txt
}

rm tmp/*
# Manual test original entities with varying seed
for i in 1 2 3 4 5 6; do
  length=$(echo "long long medium medium short short"|cut -d" " -f$i)
  min_length=$(echo "25 20 15 10 7 0"|cut -d" " -f$i)
  for s in 1; do #2 3
    seed=$(echo "10101 20202 30303"|cut -d" " -f$s)
    generate $length $min_length $seed &
  done
  #paste -d "\n" tmp/test_pred_$length.?.txt > tmp/test_pred_$length.txt
  #| sed "s/ – /–/g" |sed "s/ - /-/g" | sed "s/ — /—/g" | sed "s/( /(/g" | sed "s/ )/)/g"
  #paste -d "\n" tmp/test_scores_$length.?.txt > tmp/test_scores_$length.txt
  #paste tmp/test_scores_$length.txt tmp/test_pred_$length.txt > tmp/test_scored_pred_$length.txt
done
wait
# Join candidates and select; detokenize
paste -d "\n" tmp/scoring_*.*.txt |python select_generation.py |sed "s/ //g" |sed "s/▁/ /g" |sed "s/ – /–/g" |sed "s/ - /-/g" |sed "s/ — /—/g" |sed "s/ : /:/g" |sed "s/ \./\./g" |sed "s/ ,/,/g" |sed "s/( /(/g" |sed "s/ )/)/g" |cut -b2- > test_manual_generation.txt
#paste -d "\n" tmp/scoring_*.*.txt |python select_generation.py > test_manual_generation.txt
#paste -d "\n" data/test_manual_long.input test_manual_generation.txt /dev/null

#> all_preds_scored.txt
#"for i in 1 2 3 4 5; do
#  length=$(echo "long long medium medium short"|cut -d" " -f$i)
#  min_length=$(echo "25 20 15 10 0"|cut -d" " -f$i)
#  for s in 1 2 3; do
#    seed=$(echo "10101 20202 30303"|cut -d" " -f$s)
#    python OpenNMT-py/translate.py -seed $seed -gpu 0 -model $BEST -src data/test_manual_$length.input.pcs -output tmp/test_pred_$length.$s.txt.pcs -replace_unk -max_length 100 -batch_size $BS -min_length $min_length -length_penalty wu -alpha $ALPHA -verbose |grep "PRED SCORE"|cut -d":" -f2 > tmp/test_scores_$length.$s.txt
#    python event-augment/sentpiece/p2s.py tmp/test_pred_$length.txt.pcs event-augment/sentpiece/m.model > tmp/test_pred_$length.txt
#  done
#  cat tmp/test_pred_$length.txt | sed "s/ – /–/g" |sed "s/ - /-/g" | sed "s/ — /—/g" | sed "s/( /(/g" | sed "s/ )/)/g" > tmp/test_pred_$length.txt.detok
#  paste test_scores_$length.txt test_pred_$length.txt.detok > tmp/test_scored_pred_$length.txt.detok
#done
#paste -d "\n" data/test_manual_long.input tmp/test_scored_pred_short.txt.detok tmp/test_scored_pred_medium.txt.detok tmp/test_scored_pred_long.txt.detok /dev/null |python postprocessing.py > output/manual_eval.txt
