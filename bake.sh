rm -rf project2.zip
zip project2.zip README.md .gitignore pyproject.toml ./*.py ./*.ipynb -r ./data ./jobs ./tests ./utils ./outputs -x '**/.*' -x '**/__MACOSX'
