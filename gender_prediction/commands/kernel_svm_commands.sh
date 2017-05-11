TOP_LEVEL_DIR="${HOME}/research/gender_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/extended_indicators"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
#SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_allweek_allday.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.10%sample"
#EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_allweek_allday.csv.10%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/kernel-svm"
SOUTH_ASIAN_RESULT_PREFIX="${RESULTS_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
SOUTH_ASIAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/south_asian/test_predictions/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_RESULT_PREFIX="${RESULTS_DIR}/european/${EUROPEAN_INPUT_FILE}"
EUROPEAN_PREDICTIONS_PREFIX="${RESULTS_DIR}/european/test_predictions/${EUROPEAN_INPUT_FILE}"

# perfrom feature selection
FEATURE_SELECTION=true
# perfrom grid search?
GRID_SEARCH=true
# number of times to try?
NUM_TRIALS=3
# number of cores to use in grid search
NUM_JOBS=2

# label columns in files with all columns
EU_LABEL_COLUMN=1383
SA_LABEL_COLUMN=1257

# label columns in files with only allweek_allday columns
#EU_LABEL_COLUMN=159
#SA_LABEL_COLUMN=145

COMMON_FLAGS="-s minmax -lb 0 -ub 1" 
COMMON_SUFFIX=""
if [ "$FEATURE_SELECTION" = false ]
then
  COMMON_FLAGS="-sf $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sf"
fi

#for TRAIN_SIZE in 5000 10000 15000 20000 -1
#for TRAIN_SIZE in 5000 10000 15000 20000 30000
for TRAIN_SIZE in 5000 10000 15000 20000
do
  #for TEST_SIZE in 10000 15000
  for TEST_SIZE in 10000
  do
    for TRIAL in $(seq 1 $NUM_TRIALS)
    do
      if [ "$GRID_SEARCH" = true ]
      then
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -nj $NUM_JOBS $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX} \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}
        
        #${SCRIPTS_DIR}/run_scikit_svm.py \
        #  -trs $TRAIN_SIZE -tes $TEST_SIZE -nj $NUM_JOBS $COMMON_FLAGS -lc $SA_LABEL_COLUMN -sb \
        #  -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.sb \
        #  $SOUTH_ASIAN_INPUT_DATA \
        #  > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.sb
        
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -nj $NUM_JOBS $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX} \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}
      else
        # south asia with/without sb and cost of 1000 and 100
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 1000 -k rbf -g 0.001 -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001 \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001
        
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 1000 -k rbf -g 0.001 -lc $SA_LABEL_COLUMN -sb \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001.sb \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001.sb
        
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 10 -k rbf -g 0.01 -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01 \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01
        
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 10 -k rbf -g 0.01 -lc $SA_LABEL_COLUMN -sb \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01.sb \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01.sb
        
        # european without sb and cost of 100 and 1000
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 10 -k rbf -g 0.01 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_10.k_rbf.g_0.01
        
        ${SCRIPTS_DIR}/run_scikit_svm.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -c 1000 -k rbf -g 0.001 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.c_1000.k_rbf.g_0.001
      fi
    done
  done
done
