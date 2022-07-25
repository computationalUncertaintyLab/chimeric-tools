#!/bin/sh

git config --global user.email "git@github.com"
git config --global user.name "Github Actions CI"

git add src/chimeric_tools/data
git commit --message "[skip ci] Github Actions build: $GITHUB_RUN_NUMBER"
git fetch
git pull --rebase 
git push