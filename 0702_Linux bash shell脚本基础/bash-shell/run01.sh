#!/bin/bash
# 这是一个注释
echo "Hello World !"

echo "data and time :" $`date`

echo $(date "+%Y%m%d%H%M%S")
echo $(date "+%Y-%m-%d %H:%M:%S")

now=$(date "+%Y-%m-%d %H:%M:%S")
echo "now :" $now
today=$(date "+%Y-%m-%d")
echo "today :" $today
time=$(date "+%H:%M:%S")
echo "time :" $time
your_name="tomtom"
echo $your_name
echo "执行的文件名：$0"
#echo "第一个参数为：$1"
a=10
b=20
val=$(expr 10 + 20)
echo "a + b : $val"
echo "a + b :" $val
