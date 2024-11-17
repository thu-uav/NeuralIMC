ckpt_root=$1

ckpt_dirs=()
for dir in "$ckpt_root"/*; do
    if [ -d "$dir" ]; then
        for sub_dir in "$dir"/*; do
            if [ -d "$sub_dir" ]; then
                # Correctly escape special characters
                regular_sub_dir=$(echo "$sub_dir" | sed 's/=/\\=/g' | sed 's/,/\\,/g' | sed 's/\[/\\[/g' | sed 's/\]/\\]/g')
                ckpt_dirs+=("$regular_sub_dir")
            fi
        done
    fi
done

for dir in "${ckpt_dirs[@]}"; do
    echo "$dir"
    # Uncomment the line below to execute your python script with the corrected directory paths
    python3 eval_sim.py load_from="${dir}/wandb/latest-run/files"
    echo ">>>>"
done
