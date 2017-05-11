TOP_LEVEL_DIR="${HOME}/research/age_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
#SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_allweek_allday.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
#EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.20%sample"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_allweek_allday.csv.20%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/elastic-net"
SOUTH_ASIAN_RESULT_PREFIX="${RESULTS_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
SOUTH_ASIAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/south_asian/test_predictions/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_RESULT_PREFIX="${RESULTS_DIR}/european/${EUROPEAN_INPUT_FILE}"
EUROPEAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/european/test_predictions/${EUROPEAN_INPUT_FILE}"


# perfrom cross validation with number of alphas. If set to false, penalty and cost will be used as model
# parameters
CROSS_VALIDATION=true
NUM_ALPHAS=100
# used only if cross validation is fasle
L1_RATIO=0.99
ALPHA=0.0001

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
if [ "$CROSS_VALIDATION" = false ]
then
  COMMON_FLAGS="-sc -l $L1_RATIO -a $ALPHA $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sc.l_${L1_RATIO}.a_${ALPHA}"
else
  COMMON_FLAGS="-na $NUM_ALPHAS $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.na_${NUM_ALPHAS}"
fi

for TRAIN_SIZE in 60000
do
  for TEST_SIZE in 10000
  do
    for TRIAL in $(seq 1 $NUM_TRIALS)
    do
      # without calibration
      ${SCRIPTS_DIR}/elastic_net.py \
        -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN -nj $NUM_JOBS \
        $EUROPEAN_INPUT_DATA \
        ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX} \
        > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}
      
      # with calibration
      ${SCRIPTS_DIR}/elastic_net.py \
        -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN -nj $NUM_JOBS -c \
        $EUROPEAN_INPUT_DATA \
        ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c \
        > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c
    done
  done
done
