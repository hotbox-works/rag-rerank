rm -rf project2.zip
zip project2.zip README.md .gitignore pyproject.toml ./*.py ./*.ipynb -r ./.git ./data ./jobs ./tests ./utils -x '**/.*' -x '**/__MACOSX'
