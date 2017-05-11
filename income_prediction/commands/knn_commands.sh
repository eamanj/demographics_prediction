TOP_LEVEL_DIR="${HOME}/research/income_inequality"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
INPUT_FILE="cleaned_data_a2_c2_allweek_allday.csv"
DATA_DIR="${TOP_LEVEL_DIR}/data/bandicoot/"
INPUT_DATA="${DATA_DIR}/${INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/bandicoot/knn"
RESULT_PREFIX="${RESULTS_DIR}/${INPUT_FILE}"
PLOT_PREFIX="${RESULTS_DIR}/plots/${INPUT_FILE}"

TRAIN_SIZE=15000
TEST_SIZE=3000
NUM_JOBS=4
NUM_TRIALS=3
LABEL_COLUMN=145
declare -a EVALUATIONS=("macro-recall" "weighted-recall"
                        "macro-precision" "weighted-precision"
                        "accuracy" "mse")

for EVALUATION in ${EVALUATIONS[@]}
do
  for TRIAL in $(seq 1 $NUM_TRIALS)
  do
    # balanced data
    ${SCRIPTS_DIR}/knn.py -trs $TRAIN_SIZE -tes $TEST_SIZE \
      -s minmax -lc $LABEL_COLUMN -e $EVALUATION -nj $NUM_JOBS \
      $INPUT_DATA ${PLOT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
      > ${RESULT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
  done
done
