#!/bin/sh
# The code in this file has been adapted from the blog post https://note.com/yo4shi80/n/n620988ad8fc8.
# Removes the metadata of the notebook.
for file in $(git diff --cached --name-only); do
    if [[ $file == *.ipynb ]]; then
        # delete output
        # jupyter nbconvert --ClearOutputPreprocessor.enabled=True  \
        # --ClearOutputPreprocessor.remove_metadata_fields=[]  \
        # --ClearMetadataPreprocessor.enabled=True  \
        # --ClearMetadataPreprocessor.preserve_nb_metadata_mask='[("language_info"),("kernelspec")]' \
        # --to notebook --inplace ${file}
        # save output
        jupyter nbconvert \
        --ClearMetadataPreprocessor.enabled=True  \
        --ClearMetadataPreprocessor.preserve_nb_metadata_mask='[("language_info"),("kernelspec")]' \
        --to notebook --inplace ${file}

        jupyter nbconvert --to script $file
        git add ${file%.*}.py
        # git add .
    fi
done
