TOP_LEVEL_DIR="${HOME}/research/gender_prediction"
SCRIPTS_DIR="${TOP_LEVEL_DIR}/scripts"
DATA_DIR="${TOP_LEVEL_DIR}/data/extended_indicators"
SOUTH_ASIAN_DATA_DIR="${DATA_DIR}/south_asian"
EUROPEAN_DATA_DIR="${DATA_DIR}/european"

MISSING_SYMBOL="NA"
MIN_ACTIVE_DAYS=2
MIN_CONTACTS=2
SA_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_useful_v3_sa" 
EU_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_useful_v3_eu" 
SA_ALLWEEK_ALLDAY_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_useful_allweek_allday_v3_sa" 
EU_ALLWEEK_ALLDAY_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_useful_allweek_allday_v3_eu" 
SA_TOP200_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_top_200_columns_v3_sa"
EU_TOP200_COLUMNS_TO_REMOVE="${SCRIPTS_DIR}/data/columns_to_remove_top_200_columns_v3_eu"

# south asian
${SCRIPTS_DIR}/clean_data.py \
  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -sa -cr $SA_COLUMNS_TO_REMOVE \
  ${SOUTH_ASIAN_DATA_DIR}/indicators_attributes_joined_bangla_august2015.csv \
  ${SOUTH_ASIAN_DATA_DIR}/cleaned_south_asian_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}.csv

${SCRIPTS_DIR}/clean_data.py \
  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -sa -cr $SA_ALLWEEK_ALLDAY_COLUMNS_TO_REMOVE \
  ${SOUTH_ASIAN_DATA_DIR}/indicators_attributes_joined_bangla_august2015.csv \
  ${SOUTH_ASIAN_DATA_DIR}/cleaned_south_asian_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}_allweek_allday.csv

#${SCRIPTS_DIR}/clean_data.py \
#  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -sa -cr $SA_TOP200_COLUMNS_TO_REMOVE \
#  ${SOUTH_ASIAN_DATA_DIR}/indicators_attributes_joined_bangla_august2015.csv \
#  ${SOUTH_ASIAN_DATA_DIR}/cleaned_south_asian_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}_top200.csv

# european
${SCRIPTS_DIR}/clean_data.py \
  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -cr $EU_COLUMNS_TO_REMOVE \
  ${EUROPEAN_DATA_DIR}/indicators_attributes_joined_norway_august2015.csv \
  ${EUROPEAN_DATA_DIR}/cleaned_european_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}.csv

${SCRIPTS_DIR}/clean_data.py \
  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -cr $EU_ALLWEEK_ALLDAY_COLUMNS_TO_REMOVE \
  ${EUROPEAN_DATA_DIR}/indicators_attributes_joined_norway_august2015.csv \
  ${EUROPEAN_DATA_DIR}/cleaned_european_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}_allweek_allday.csv

#${SCRIPTS_DIR}/clean_data.py \
#  -n $MISSING_SYMBOL -a $MIN_ACTIVE_DAYS -c $MIN_CONTACTS -cr $EU_TOP200_COLUMNS_TO_REMOVE \
#  ${EUROPEAN_DATA_DIR}/indicators_attributes_joined_norway_august2015.csv \
#  ${EUROPEAN_DATA_DIR}/cleaned_european_data_a${MIN_ACTIVE_DAYS}_c${MIN_CONTACTS}_top200.csv
