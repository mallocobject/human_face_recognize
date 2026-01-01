#!/bin/bash

# 设置文件夹路径
SOURCE_DIR="YaleB/yaleB03"
TRAIN_DIR="$SOURCE_DIR/train"
TEST_DIR="$SOURCE_DIR/test"

# 确保train和test文件夹存在，不存在则创建
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

FILES=($(ls "$SOURCE_DIR" | grep -vE '^train$|^test$' | shuf))

# 确保只处理前41个文件
FILES=("${FILES[@]:0:41}")

# 将前30个文件移动到train文件夹，并重命名为0, 1, ..., 29
for i in $(seq 0 29); do
    mv "$SOURCE_DIR/${FILES[$i]}" "$TRAIN_DIR/$i"
done

# 将接下来的10个文件移动到test文件夹，并重命名为30, 31, ..., 39
for i in $(seq 30 40); do
    mv "$SOURCE_DIR/${FILES[$i]}" "$TEST_DIR/$i"
done

echo "Files have been split into 'train' and 'test' folders."
