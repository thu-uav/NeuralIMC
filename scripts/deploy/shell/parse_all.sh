data_root=$1

file_path=()
for file in $data_root/*; do
    if [ -f "$file" ]; then
        # regular_path=$(echo $file | sed 's/=/\\=/g' | sed 's/,/\\,/g')
        file_path+=("${data_root}/${file##*/}")
    fi
done

for file in "${file_path[@]}"; do
    echo "processing ${file}..."
    python3 parse_logs.py --log_name="${file}"
    echo ""
done
