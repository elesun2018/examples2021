#!/bin/bash
#在不同时间段，执行对应任务
#https://blog.csdn.net/qq_41204464/article/details/105712054
while true;do
    out_time=`date '+%Y-%m-%d-%H:%M'`  #格式：2019-04-24-21:26
    echo "$out_time"
    #获取当前时间，格式是时分，例如当前是上午8：50，hh=850
    hh=`date '+%H%M'`
    #早上7.30--7.45 执行自动做早餐的任务
    if [ $hh -ge 730 -a $hh -le 745 ]
    then
        echo " Morning -- Automatic breakfast "
    #中午11.52--12.15 执行做饭任务
    elif [ $hh -ge 1152 -a $hh -le 1215 ]
    then
        echo " Lunch time -- Cook "
    #下午17:23--17.40 执行自动浇花任务
    elif [ $hh -ge 1723 -a $hh -le 1740 ]
    then
        echo "night -- Automatic watering"
    #不适合适的时间，不做什么
    else
        echo "$hh Not within time "
    fi
    sleep 60 #休息5s
done
