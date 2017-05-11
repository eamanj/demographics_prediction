TOP_LEVEL_DIR="${HOME}/research/income_inequality"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
INPUT_FILE="cleaned_data_a2_c2_allweek_allday.csv"
DATA_DIR="${TOP_LEVEL_DIR}/data/bandicoot/"
INPUT_DATA="${DATA_DIR}/${INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/bandicoot/ordinal-logistic"
RESULT_PREFIX="${RESULTS_DIR}/${INPUT_FILE}"
PLOT_PREFIX="${RESULTS_DIR}/plots/${INPUT_FILE}"

TRAIN_SIZE=15000
TEST_SIZE=3000
NUM_JOBS=4
NUM_TRIALS=3
LABEL_COLUMN=145
declare -a EVALUATIONS=("macro-recall" "weighted-recall"
                        "macro-precision" "weighted-precision"
                        "accuracy")


# these algorithms do not support changing the penalties to account for imabalacned data.
# so we only run them with abalanced data.
for ALGORITHM in "lad" "ordinalridge" "logisticit" "logisticat" "logisticse"
do
  for EVALUATION in ${EVALUATIONS[@]}
  do
    for TRIAL in $(seq 1 $NUM_TRIALS)
    do
      ${SCRIPTS_DIR}/ordinal_logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
        -lc $LABEL_COLUMN -oa $ALGORITHM -e $EVALUATION -nj $NUM_JOBS \
        $INPUT_DATA ${PLOT_PREFIX}.oa_${ALGORITHM}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
        > ${RESULT_PREFIX}.oa_${ALGORITHM}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
    done
  done
done

# with scaling
for ALGORITHM in "lad" "ordinalridge" "logisticit" "logisticat" "logisticse"
do
  for EVALUATION in ${EVALUATIONS[@]}
  do
    for TRIAL in $(seq 1 $NUM_TRIALS)
    do
      ${SCRIPTS_DIR}/ordinal_logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
        -s standard -lc $LABEL_COLUMN -oa $ALGORITHM -e $EVALUATION -nj $NUM_JOBS \
        $INPUT_DATA ${PLOT_PREFIX}.s_standard.oa_${ALGORITHM}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
        > ${RESULT_PREFIX}s_standard.oa_${ALGORITHM}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
    done
  done
done
