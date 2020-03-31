outmax=100
BEST=trained_model.pt
BS=128
ALPHA=1.6

rm tmp/scoring*
# Manual test original entities with varying seed
for i in 1 2 3 4 5 6; do
  length=$(echo "long long medium medium short short"|cut -d" " -f$i)
  min_length=$(echo "25 20 15 10 7 0"|cut -d" " -f$i)
  for s in 1; do #2 3
    seed=$(echo "10101 20202 30303"|cut -d" " -f$s)
    python OpenNMT-py/translate.py -seed $seed -gpu 0 -model $BEST -src data/devel.input.pcs_$length -output tmp/test_pred_$min_length.$s.txt -replace_unk -max_length $outmax -batch_size $BS -min_length $min_length -length_penalty wu -alpha $ALPHA -verbose -attn_debug -beam_size 15 > tmp/trans.output
    #grep "PRED SCORE" trans.output | cut -d":" -f2 > tmp/test_scores_$length.$s.txt
    python calc_attn_errors.py -i tmp/trans.output -o tmp/scoring_$min_length.$s.txt
    grep -n "" tmp/scoring_$min_length.$s.txt |sed -E "s/(^[0-9]+):/\1\t/g" |while read LINE;do echo -e "$length\t$min_length\t$LINE";done |sed "s/ //g" |sed "s/▁/ /g" |cut -b3- > tmp/scoring_$min_length.$s.txt_
    #cat tmp/scoring_$min_length.$s.txt |while read LINE;do echo -e "$length\t$min_length\t$LINE";done > tmp/scoring_$min_length.$s.txt_
    mv tmp/scoring_$min_length.$s.txt_ tmp/scoring_$min_length.$s.txt
    head tmp/scoring_$min_length.$s.txt
  done
  #paste -d "\n" tmp/test_pred_$length.?.txt > tmp/test_pred_$length.txt
  #| sed "s/ – /–/g" |sed "s/ - /-/g" | sed "s/ — /—/g" | sed "s/( /(/g" | sed "s/ )/)/g"
  #paste -d "\n" tmp/test_scores_$length.?.txt > tmp/test_scores_$length.txt
  #paste tmp/test_scores_$length.txt tmp/test_pred_$length.txt > tmp/test_scored_pred_$length.txt
done
paste -d "\n" tmp/scoring_*.*.txt > devel_all_generations.txt
#|python select_generation.py > test_manual_generation.txt
#paste -d "\n" data/test_manual_long.input test_manual_generation.txt /dev/null

# Get examples for evaluation
#paste -d "\n" data/devel.input tmp/scoring_*.*.txt|tail -n +210|head -210

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
