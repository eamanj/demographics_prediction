TOP_LEVEL_DIR="${HOME}/research/income_inequality"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts/prediction"
INPUT_FILE="cleaned_data_a2_c2_allweek_allday.csv"
#INPUT_FILE="cleaned_data_a2_c2.csv"
DATA_DIR="${TOP_LEVEL_DIR}/data/bandicoot/"
INPUT_DATA="${DATA_DIR}/${INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/bandicoot/logistic"
RESULT_PREFIX="${RESULTS_DIR}/${INPUT_FILE}"
PLOT_PREFIX="${RESULTS_DIR}/plots/${INPUT_FILE}"

NUM_JOBS=2
NUM_TRIALS=2
LABEL_COLUMN=145
#LABEL_COLUMN=1257
#declare -a EVALUATIONS=("macro-recall" "weighted-recall"
#                        "macro-precision" "weighted-precision"
#                        "accuracy" "mse")
declare -a EVALUATIONS=("accuracy")

# balanced data
TRAIN_SIZE=15000
TEST_SIZE=3000
for EVALUATION in ${EVALUATIONS[@]}
do
  for TRIAL in $(seq 1 $NUM_TRIALS)
  do
    ${SCRIPTS_DIR}/logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
      -lc $LABEL_COLUMN -e $EVALUATION -nj $NUM_JOBS \
      $INPUT_DATA ${PLOT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
      > ${RESULT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
  done
done

TRAIN_SIZE=20000
TEST_SIZE=3000
for EVALUATION in ${EVALUATIONS[@]}
do
  for TRIAL in $(seq 1 $NUM_TRIALS)
  do
    ${SCRIPTS_DIR}/logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
      -lc $LABEL_COLUMN -e $EVALUATION -nj $NUM_JOBS \
      $INPUT_DATA ${PLOT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
      > ${RESULT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
  done
done

TRAIN_SIZE=25000
TEST_SIZE=3000
for EVALUATION in ${EVALUATIONS[@]}
do
  for TRIAL in $(seq 1 $NUM_TRIALS)
  do
    ${SCRIPTS_DIR}/logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
      -lc $LABEL_COLUMN -e $EVALUATION -nj $NUM_JOBS \
      $INPUT_DATA ${PLOT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE} \
      > ${RESULT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}
  done
done


# imbalanced data
TRAIN_SIZE=30000
TEST_SIZE=3000
for EVALUATION in ${EVALUATIONS[@]}
do
  for TRIAL in $(seq 1 $NUM_TRIALS)
  do
    ${SCRIPTS_DIR}/logistic.py  -trs $TRAIN_SIZE -tes $TEST_SIZE \
      -lc $LABEL_COLUMN -e $EVALUATION -nj $NUM_JOBS -i \
      $INPUT_DATA ${PLOT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}.i \
      > ${RESULT_PREFIX}.e_${EVALUATION}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}.i
  done
done
