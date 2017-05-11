TOP_LEVEL_DIR="${HOME}/research/age_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
#SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_allweek_allday.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.20%sample"
#EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_allweek_allday.csv.20%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

RESULTS_DIR="${TOP_LEVEL_DIR}/results/logistic"
SOUTH_ASIAN_RESULT_PREFIX="${RESULTS_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
SOUTH_ASIAN_PLOTS_PREFIX="${RESULTS_DIR}/south_asian/plots/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_RESULT_PREFIX="${RESULTS_DIR}/european/${EUROPEAN_INPUT_FILE}"
EUROPEAN_PLOTS_PREFIX="${RESULTS_DIR}/european/plots/${EUROPEAN_INPUT_FILE}"

# perfrom feature selection. set FS_Cost to skip cross validatoin of l1-svm feature
# selector and directly set the cost parameter.
# feature selection
FEATURE_SELECTION=false
FS_COST=""
THRESHOLDS=( 0 0.1 0.2 0.3 0.4 0.5 0.6 )

# perfrom cross validation with number of alphas. If set to false, uses the cost parameter
# here.
# parameters
CROSS_VALIDATION=true
COST=""

# number of times to try?
NUM_TRIALS=2
# number of cores to use in grid search
NUM_JOBS=20

# label columns in files with all columns
EU_LABEL_COLUMN=1384
SA_LABEL_COLUMN=1258

# label columns in files with only allweek_allday columns
#EU_LABEL_COLUMN=160
#SA_LABEL_COLUMN=146

# number of equal sized classes to generate from age data
NUM_CLASSES=4
# The criteria to opitmize in Cross-validation of logisitc
CRITERIA="accuracy"

COMMON_FLAGS="-nc $NUM_CLASSES"
COMMON_SUFFIX=".nc_${NUM_CLASSES}"
if [ "$FEATURE_SELECTION" = false ]
then
  THRESHOLDS=( 0 )
  COMMON_FLAGS="-sf $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sf"
elif [ -n "$FS_COST" ]
then
  COMMON_FLAGS="-fc $FS_COST $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.fc_${FS_COST}"
fi

if [ "$CROSS_VALIDATION" = false ]
then
  COMMON_FLAGS="-sc -c $COST $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sc.c_${COST}"
else
  COMMON_FLAGS="-e $CRITERIA $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.e_${CRITERIA}"
fi


# european
for TRIAL in $(seq 1 $NUM_TRIALS)
do
  for TRAIN_SIZE in 25000 
  do
    for TEST_SIZE in 10000
    do
      for MULTI_CLASS in "ovr"
      do
        # multinomial is only supported with l2 penalty
        PENALTIES=( "l1" )

        for PENALTY in "${PENALTIES[@]}"
        do
          for THRESHOLD in "${THRESHOLDS[@]}"
          do
            ${SCRIPTS_DIR}/logistic.py \
              -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
              -p $PENALTY -m $MULTI_CLASS -t $THRESHOLD -nj $NUM_JOBS \
              $EUROPEAN_INPUT_DATA \
              ${EUROPEAN_PLOTS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.p_${PENALTY}.m_${MULTI_CLASS}.t_${THRESHOLD} \
              > ${EUROPEAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.p_${PENALTY}.m_${MULTI_CLASS}.t_${THRESHOLD}
          done 
        done
      done
    done
  done
done


# south asian 
for TRIAL in $(seq 1 $NUM_TRIALS)
do
  for TRAIN_SIZE in 25000 
  do
    for TEST_SIZE in 5000
    do
      for MULTI_CLASS in "ovr"
      do
        # multinomial is only supported with l2 penalty
        PENALTIES=( "l1" )

        for PENALTY in "${PENALTIES[@]}"
        do
          for THRESHOLD in "${THRESHOLDS[@]}"
          do
            ${SCRIPTS_DIR}/logistic.py \
              -trs $TRAIN_SIZE -tes $TEST_SIZE $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
              -p $PENALTY -m $MULTI_CLASS -t $THRESHOLD -nj $NUM_JOBS \
              $SOUTH_ASIAN_INPUT_DATA \
              ${SOUTH_ASIAN_PLOTS_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.p_${PENALTY}.m_${MULTI_CLASS}.t_${THRESHOLD} \
              > ${SOUTH_ASIAN_RESULT_PREFIX}.trial_${TRIAL}.trs_${TRAIN_SIZE}.tes_${TEST_SIZE}${COMMON_SUFFIX}.p_${PENALTY}.m_${MULTI_CLASS}.t_${THRESHOLD}
          done 
        done
      done
    done
  done
done
