TOP_LEVEL_DIR="${HOME}/research/gender_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"

# inputs
DATA_DIR="${TOP_LEVEL_DIR}/data/extended_indicators"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2.csv.20%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

# input file prefix per country (eu or sa) and per algorithm
EU_RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/train-size/european"
EU_LINEAR_PREFIX="${EU_RESULTS_DIR}/linear-svm/${EUROPEAN_INPUT_FILE}"
EU_KERNEL_PREFIX="${EU_RESULTS_DIR}/kernel-svm/${EUROPEAN_INPUT_FILE}"
EU_RF_PREFIX="${EU_RESULTS_DIR}/random-forest/${EUROPEAN_INPUT_FILE}"
EU_LOGISTIC_PREFIX="${EU_RESULTS_DIR}/logistic/${EUROPEAN_INPUT_FILE}"
EU_KNN_PREFIX="${EU_RESULTS_DIR}/knn/${EUROPEAN_INPUT_FILE}"

SA_RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/train-size/south_asian"
SA_LINEAR_PREFIX="${SA_RESULTS_DIR}/linear-svm/${SOUTH_ASIAN_INPUT_FILE}"
SA_KERNEL_PREFIX="${SA_RESULTS_DIR}/kernel-svm/${SOUTH_ASIAN_INPUT_FILE}"
SA_RF_PREFIX="${SA_RESULTS_DIR}/random-forest/${SOUTH_ASIAN_INPUT_FILE}"
SA_LOGISTIC_PREFIX="${SA_RESULTS_DIR}/logistic/${SOUTH_ASIAN_INPUT_FILE}"
SA_KNN_PREFIX="${SA_RESULTS_DIR}/knn/${SOUTH_ASIAN_INPUT_FILE}"

# perfrom feature selection
FEATURE_SELECTION=true
# perfrom grid search?
GRID_SEARCH=true
NUM_TRIALS=8
NUM_PROCESSES=8
TEST_SIZE=10000

EU_LABEL_COLUMN=1383
SA_LABEL_COLUMN=1257

COMMON_FLAGS="" 
COMMON_SUFFIX=""
if [ "$FEATURE_SELECTION" = false ]
then
  COMMON_FLAGS="-sfs $COMMON_FLAGS"
  COMMON_SUFFIX="${COMMON_SUFFIX}.sfs"
fi

dry_mode=false

if [ "$dry_mode" = true ]; then
  if [ "$GRID_SEARCH" = true ]
  then
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -sk linear -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""

    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -sk linear -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}"
    echo -e ""

  else
    # european
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1 -sk linear -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1.sk_linear${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1000 -sk rbf -sg 0.001 -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1000.sk_rbf.sg_0.001${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -rnj 1 -rnt 400 -rc entropy -rmf sqrt -rmss 5 -rmsl 2 -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.rnt_400.rc_entropy.rmf_sqrt.rmss_5.rmsl_2${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -lop l1 -loc 1 -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.lop_l1.loc_1${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -knn 70 -kw distance -ka ball_tree -km manhattan -lc $EU_LABEL_COLUMN \\
      $EUROPEAN_INPUT_DATA ${EU_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.knn_70.kw_distance.ka_ball_tree.km_manhattan${COMMON_SUFFIX}"
    echo -e ""

    # south asian
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 10 -sk linear -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_10.sk_linear${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1000 -sk rbf -sg 0.001 -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1000.sk_rbf.sg_0.001${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -rnj 1 -rnt 800 -rc entropy -rmf 0.7 -rmss 20 -rmsl 10 -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.rnt_800.rc_entropy.rmf_0.7.rmss_20.rmsl_10${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -lop l1 -loc 1 -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.lop_l1.loc_1${COMMON_SUFFIX}"
    echo -e ""
    
    echo "${SCRIPTS_DIR}/vary_train_size.py \\
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -knn 150 -kw distance -ka ball_tree -km manhattan -lc $SA_LABEL_COLUMN \\
      $SOUTH_ASIAN_INPUT_DATA ${SA_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.knn_150.kw_distance.ka_ball_tree.km_manhattan${COMMON_SUFFIX}"
    echo -e ""
  fi

else
  if [ "$GRID_SEARCH" = true ]
  then
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -sk linear -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}

    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -sk linear -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}${COMMON_SUFFIX}

  else
    # european
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1 -sk linear -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1.sk_linear${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1000 -sk rbf -sg 0.001 -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1000.sk_rbf.sg_0.001${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -rnj 1 -rnt 400 -rc entropy -rmf sqrt -rmss 5 -rmsl 2 -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.rnt_400.rc_entropy.rmf_sqrt.rmss_5.rmsl_2${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -lop l1 -loc 1 -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.lop_l1.loc_1${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -knn 70 -kw distance -ka ball_tree -km manhattan -lc $EU_LABEL_COLUMN \
      $EUROPEAN_INPUT_DATA ${EU_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.knn_70.kw_distance.ka_ball_tree.km_manhattan${COMMON_SUFFIX}

    # south asian
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 10 -sk linear -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_LINEAR_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_10.sk_linear${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la svm -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -sc 1000 -sk rbf -sg 0.001 -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_KERNEL_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.sc_1000.sk_rbf.sg_0.001${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la random-forest -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -rnj 1 -rnt 800 -rc entropy -rmf 0.7 -rmss 20 -rmsl 10 -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_RF_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.rnt_800.rc_entropy.rmf_0.7.rmss_20.rmsl_10${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la logistic -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -lop l1 -loc 1 -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.lop_l1.loc_1${COMMON_SUFFIX}
    
    ${SCRIPTS_DIR}/vary_train_size.py \
      -ts $TEST_SIZE -la knn -nt $NUM_TRIALS -np $NUM_PROCESSES $COMMON_FLAGS -sgs -knn 150 -kw distance -ka ball_tree -km manhattan -lc $SA_LABEL_COLUMN \
      $SOUTH_ASIAN_INPUT_DATA ${SA_KNN_PREFIX}.ts_${TEST_SIZE}.nt_${NUM_TRIALS}.knn_150.kw_distance.ka_ball_tree.km_manhattan${COMMON_SUFFIX}
  fi
fi
