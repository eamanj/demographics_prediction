TOP_LEVEL_DIR="${HOME}/research/age_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
#SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_allweek_allday.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
#EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.20%sample"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_allweek_allday.csv.20%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/linear"
SOUTH_ASIAN_RESULT_PREFIX="${RESULTS_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
SOUTH_ASIAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/south_asian/test_predictions/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_RESULT_PREFIX="${RESULTS_DIR}/european/${EUROPEAN_INPUT_FILE}"
EUROPEAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/european/test_predictions/${EUROPEAN_INPUT_FILE}"

# perfrom feature selection. set l1_ratio and alpha if you don't want to cross validate
# feature selection
FEATURE_SELECTION=true
#L1_RATO=0.99
#ALPHA=0.0001
L1_RATIO=""
ALPHA=""
THRESHOLDS=( 0 0.1 0.2 0.3 0.4 0.5 0.6 )

# number of times to try?
NUM_TRIALS=2
# number of cores to use in grid search
NUM_JOBS=3

# label columns in files with all columns
#EU_LABEL_COLUMN=1384
#SA_LABEL_COLUMN=1258

# label columns in files with only allweek_allday columns
EU_LABEL_COLUMN=160
SA_LABEL_COLUMN=146

COMMON_FLAGS=""
COMMON_SUFFIX=""
if [ "$FEATURE_SELECTION" = false ]
then
  THRESHOLDS=( 0 )
  COMMON_FLAGS="-sf $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sf"
elif [[ -n "$L1_RATIO" && -n "$ALPHA" ]]
then
  COMMON_FLAGS="-l $L1_RATIO -a $ALPHA $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.l_${L1_RATIO}.a_${ALPHA}"
fi

for TRAIN_SIZE in 15000 30000
do
  for TEST_SIZE in 10000
  do
    for TRIAL in $(seq 1 $NUM_TRIALS)
    do
      for WEIGHT in "uniform" "linear" "square" "freq" "freq-square"
      do
        for THRESHOLD in "${THRESHOLDS[@]}"
        do
          # without calibration
          ${SCRIPTS_DIR}/linear.py \
            -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
            -w $WEIGHT -t $THRESHOLD -nj $NUM_JOBS \
            $EUROPEAN_INPUT_DATA \
            ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.w_${WEIGHT}.t_${THRESHOLD} \
            > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.w_${WEIGHT}.t_${THRESHOLD}

          # with calibration
          ${SCRIPTS_DIR}/linear.py \
            -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
            -w $WEIGHT -t $THRESHOLD -nj $NUM_JOBS -c \
            $EUROPEAN_INPUT_DATA \
            ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.w_${WEIGHT}.t_${THRESHOLD}.c \
            > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.w_${WEIGHT}.t_${THRESHOLD}.c
        done 
      done
    done
  done
done
