TOP_LEVEL_DIR="${HOME}/research/gender_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"

# inputs
DATA_DIR="${TOP_LEVEL_DIR}/data/extended_indicators"
SOUTH_ASIAN_INPUT_FILE="cleaned_south_asian_data_a2_c2_top200.csv"
SOUTH_ASIAN_INPUT_DATA="${DATA_DIR}/south_asian/${SOUTH_ASIAN_INPUT_FILE}"
EUROPEAN_INPUT_FILE="cleaned_european_data_a2_c2_top200.csv.20%sample"
EUROPEAN_INPUT_DATA="${DATA_DIR}/european/${EUROPEAN_INPUT_FILE}"

# input file prefix per country (eu or sa) and per algorithm
EU_RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/feature-selection-l1-svm/european"
EU_LINEAR_PREFIX="${EU_RESULTS_DIR}/linear-svm/${EUROPEAN_INPUT_FILE}"
EU_KERNEL_PREFIX="${EU_RESULTS_DIR}/kernel-svm/${EUROPEAN_INPUT_FILE}"
EU_RF_PREFIX="${EU_RESULTS_DIR}/random-forest/${EUROPEAN_INPUT_FILE}"
EU_LOGISTIC_PREFIX="${EU_RESULTS_DIR}/logistic/${EUROPEAN_INPUT_FILE}"
EU_KNN_PREFIX="${EU_RESULTS_DIR}/knn/${EUROPEAN_INPUT_FILE}"

SA_RESULTS_DIR="${TOP_LEVEL_DIR}/results/extended_indicators/feature-selection-l1-svm/south_asian"
SA_LINEAR_PREFIX="${SA_RESULTS_DIR}/linear-svm/${SOUTH_ASIAN_INPUT_FILE}"
SA_KERNEL_PREFIX="${SA_RESULTS_DIR}/kernel-svm/${SOUTH_ASIAN_INPUT_FILE}"
SA_RF_PREFIX="${SA_RESULTS_DIR}/random-forest/${SOUTH_ASIAN_INPUT_FILE}"
SA_LOGISTIC_PREFIX="${SA_RESULTS_DIR}/logistic/${SOUTH_ASIAN_INPUT_FILE}"
SA_KNN_PREFIX="${SA_RESULTS_DIR}/knn/${SOUTH_ASIAN_INPUT_FILE}"

MIN_NF=1
MAX_NF=150
TEST_SIZE=10000
SA_NUM_SAMPLES=40000
EU_NUM_SAMPLES=70000
LABEL_COLUMN=200

# south asian
${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la svm -c 10 -k linear -lc $LABEL_COLUMN -ns $SA_NUM_SAMPLES \
  $SOUTH_ASIAN_INPUT_DATA \
  ${SA_LINEAR_PREFIX}.ts_${TEST_SIZE}.c_10.k_linear \
  | tee ${SA_LINEAR_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la svm -c 1000 -k rbf -g 0.001 -lc $LABEL_COLUMN -ns $SA_NUM_SAMPLES \
  $SOUTH_ASIAN_INPUT_DATA \
  ${SA_KERNEL_PREFIX}.ts_${TEST_SIZE}.c_1000.k_rbf.g_0.001 \
  | tee ${SA_KERNEL_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la random-forest -nt 400 -cr entropy -mf 0.7 -mss 20 -msl 10 -lc $LABEL_COLUMN -ns $SA_NUM_SAMPLES \
  $SOUTH_ASIAN_INPUT_DATA \
  ${SA_RF_PREFIX}.ts_${TEST_SIZE}.nt_400.cr_entropy.mf_0.7.mss_20.msl_10 \
  | tee ${SA_RF_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la logistic -lp l1 -lco 1 -lc $LABEL_COLUMN -ns $SA_NUM_SAMPLES \
  $SOUTH_ASIAN_INPUT_DATA \
  ${SA_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.lp_l1.lco_1 \
  | tee ${SA_LOGISTIC_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la knn -nn 150 -w distance -a ball_tree -m manhattan -lc $LABEL_COLUMN -ns $SA_NUM_SAMPLES \
  $SOUTH_ASIAN_INPUT_DATA \
  ${SA_KNN_PREFIX}.ts_${TEST_SIZE}.nn_150.w_distance.a_ball_tree.m_manhattan \
  | tee ${SA_KNN_PREFIX}.features_selection_output


# european
${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la svm -c 1 -k linear -lc $LABEL_COLUMN -ns $EU_NUM_SAMPLES \
  $EUROPEAN_INPUT_DATA \
  ${EU_LINEAR_PREFIX}.ts_${TEST_SIZE}.c_1.k_linear \
  | tee ${EU_LINEAR_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la svm -c 1000 -k rbf -g 0.001 -lc $LABEL_COLUMN -ns $EU_NUM_SAMPLES \
  $EUROPEAN_INPUT_DATA \
  ${EU_KERNEL_PREFIX}.ts_${TEST_SIZE}.c_1000.k_rbf.g_0.001 \
  | tee ${EU_KERNEL_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la random-forest -nt 400 -cr entropy -mf sqrt -mss 5 -msl 2 -lc $LABEL_COLUMN -ns $EU_NUM_SAMPLES \
  $EUROPEAN_INPUT_DATA \
  ${EU_RF_PREFIX}.ts_${TEST_SIZE}.nt_400.cr_entropy.mf_sqrt.mss_5.msl_2 \
  | tee ${EU_RF_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la logistic -lp l1 -lco 1 -lc $LABEL_COLUMN -ns $EU_NUM_SAMPLES \
  $EUROPEAN_INPUT_DATA \
  ${EU_LOGISTIC_PREFIX}.ts_${TEST_SIZE}.lp_l1.lco_1 \
  | tee ${EU_LOGISTIC_PREFIX}.features_selection_output

${SCRIPTS_DIR}/feature_selection_l1_svm.py \
  -min_nf $MIN_NF -max_nf $MAX_NF -ts $TEST_SIZE -la knn -nn 70 -w distance -a ball_tree -m manhattan -lc $LABEL_COLUMN -ns $EU_NUM_SAMPLES \
  $EUROPEAN_INPUT_DATA \
  ${EU_KNN_PREFIX}.ts_${TEST_SIZE}.nn_70.w_distance.a_ball_tree.m_manhattan \
  | tee ${EU_KNN_PREFIX}.features_selection_output
