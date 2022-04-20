#!/bin/bash
#在一个时间周期内运行shell脚本
#https://blog.csdn.net/z1164072826/article/details/80180333
begin=$1
end=$2
echo "begin $begin"
echo "end $end"
be_s="${begin:0:8} ${begin:8:2}"
en_s="${end:0:8} ${end:8:2}"
echo "be_s $be_s"
echo "en_s $en_s"
be_s=$(date -d "${be_s}" +%s)
en_s=$(date -d "${en_s}" +%s)
echo "be_s $be_s"
echo "en_s $en_s"
while [ "$be_s" -le "$en_s" ]
do
        datefmt=`date -d "1970-01-01 UTC ${be_s} seconds" +%Y%m%d%H` #datefmt为天级日期
        echo ${datefmt}
        #添加处理代码，日期为${datefmt}
        sh writedate2log.sh ${datefmt}
        be_s=$((be_s+3600))
done
