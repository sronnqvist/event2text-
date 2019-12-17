outmax=100
BEST=trained_model.pt
BS=128
ALPHA=1.8

# Manual test original entities with varying seed
for i in 1 2 3 4 5; do
  length=$(echo "long long medium medium short"|cut -d" " -f$i)
  min_length=$(echo "25 20 15 10 0"|cut -d" " -f$i)
  for s in 1 2 3; do
    seed=$(echo "10101 20202 30303"|cut -d" " -f$s)
    python OpenNMT-py/translate.py -seed $seed -gpu 0 -model $BEST -src data/test_manual_$length.input -output tmp/test_pred_$length.$s.txt -replace_unk -max_length $outmax -batch_size $BS -min_length $min_length -length_penalty wu -alpha $ALPHA -verbose -attn_debug > tmp/trans.output
    #grep "PRED SCORE" trans.output | cut -d":" -f2 > tmp/test_scores_$length.$s.txt
    python calc_attn_errors.py -i tmp/trans.output -o tmp/scoring_$length.$s.txt
  done
  #paste -d "\n" tmp/test_pred_$length.?.txt > tmp/test_pred_$length.txt
  #| sed "s/ – /–/g" |sed "s/ - /-/g" | sed "s/ — /—/g" | sed "s/( /(/g" | sed "s/ )/)/g"
  #paste -d "\n" tmp/test_scores_$length.?.txt > tmp/test_scores_$length.txt
  #paste tmp/test_scores_$length.txt tmp/test_pred_$length.txt > tmp/test_scored_pred_$length.txt
done
paste -d "\n" tmp/scoring_*.*.txt |less
#> all_preds_scored.txt
