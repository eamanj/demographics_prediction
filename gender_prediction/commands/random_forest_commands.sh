TOP_LEVEL_DIR="${HOME}/research/gender_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/extended_indicators"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
#SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_allweek_allday.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.10%sample"
#EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_allweek_allday.csv.10%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/random-forest"
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

COMMON_FLAGS="-nj $NUM_JOBS"
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
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX} \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}
        
        #${SCRIPTS_DIR}/run_scikit_random_forest.py \
        #  -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $SA_LABEL_COLUMN -sb \
        #  -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.sb \
        #  $SOUTH_ASIAN_INPUT_DATA \
        #  > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.sb
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX} \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}
      else
        # south asia with/without sb and 800 trees
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf 0.5 -mss 5 -msl 1 -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.5.mss_5.msl_1 \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.5.mss_5.msl_1
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf 0.7 -mss 20 -msl 10 -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.7.mss_20.msl_10 \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.7.mss_20.msl_10
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf 0.5 -mss 5 -msl 1 -sb -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.5.mss_5.msl_1.sb \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.5.mss_5.msl_1.sb
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf 0.7 -mss 20 -msl 10 -sb -lc $SA_LABEL_COLUMN \
          -o ${SOUTH_ASIAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.7.mss_20.msl_10.sb \
          $SOUTH_ASIAN_INPUT_DATA \
          > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.7.mss_20.msl_10.sb
        
        # european without sb and 800/400 trees
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf sqrt -mss 5 -msl 2 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_sqrt.mss_5.msl_2 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_sqrt.mss_5.msl_2
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 400 -c entropy -mf sqrt -mss 5 -msl 2 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_400.c_entropy.mf_sqrt.mss_5.msl_2 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_400.c_entropy.mf_sqrt.mss_5.msl_2
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 800 -c entropy -mf 0.9 -mss 2 -msl 1 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.9.mss_2.msl_1 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_800.c_entropy.mf_0.9.mss_2.msl_1
        
        ${SCRIPTS_DIR}/run_scikit_random_forest.py \
          -trs $TRAIN_SIZE -tes $TEST_SIZE -sg $COMMON_FLAGS -nt 400 -c entropy -mf 0.9 -mss 2 -msl 1 -lc $EU_LABEL_COLUMN \
          -o ${EUROPEAN_PREDICTIONS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_400.c_entropy.mf_0.9.mss_2.msl_1 \
          $EUROPEAN_INPUT_DATA \
          > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.nt_400.c_entropy.mf_0.9.mss_2.msl_1
      fi
    done
  done
done
