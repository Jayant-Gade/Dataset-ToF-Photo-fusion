find gesturanet_augmented_dataset/ -type f \( -name "img_*_rot*.jpg" -o -name "img_*_rot*.npy" \) | wc -l

find gesturanet_augmented_dataset/ -type f \( -name "img_*_orig_*.jpg" -o -name "img_*_orig_*.npy" -o -name "img_*_org_*.jpg" -o -name "img_*_org_*.npy" \) | wc -l

replace (| wc -l with -delete)