INPUT_DIR=""
while test $# -gt 0; do
  case "$1" in
    -i)
      shift
      if test $# -gt 0; then
        INPUT_DIR=$1
      else
        echo "extract_feature_selection_l1_results.sh -i input_dir"
        echo "no input dir specified"
        exit 1
      fi
      shift
      ;;
    *)
      break ;;
  esac
done

if [[ -z $INPUT_DIR ]]; then 
  echo "extract_feature_selection_l1_results.sh -i input_dir"
  echo "no input dir specified"
  exit 1
fi

NUM_FILES=`find $INPUT_DIR -type f | wc -l`

echo "num selected features,test accuracy"
for f in $INPUT_DIR/*
do
  NUM_SELECTED_FEATURES=`grep "Out of" $f | cut -d ' ' -f 5`
  ACCURACY=`grep "Test Accuracy" $f | cut -d ' ' -f 3`
  echo "$NUM_SELECTED_FEATURES,$ACCURACY"
done |
sort -t ',' -n -k 1,1 -u
