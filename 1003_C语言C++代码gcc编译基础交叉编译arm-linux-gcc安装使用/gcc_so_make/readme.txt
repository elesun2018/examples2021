linux 下生成动态库.so并引用
https://blog.csdn.net/junzhang1122/article/details/42048599?utm_medium=distribute.wap_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.wap_blog_relevant_pic&depth_1-utm_source=distribute.wap_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.wap_blog_relevant_pic
动态库的引入及减少了主代码文件的大小，同时生成的动态库又是动态加载的，只有运行的时候才去加载，linux 下的 动态库 .so 就像windows下的 DLL一样。

注意：
生成的 xx.so文件要添加他的路径，否则加载不上

用 export LD_LIBRARY_PATH=./  或者在 /etc/ld.so.conf中加入xxx.so所在的目录，然后/sbin/ldconfig –v更新一下配置即可。

Makfile中两个变量引用 ${xx_1}${xx_2}，在 xx_1后面一定不要有空格，否则引用变量的时候会把该空格引入其中，但是做这个例子的时候就遇到过这个问题，后来查询半天才发现，希望引以为戒。
————————————————
版权声明：本文为CSDN博主「junzhang1122」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/junzhang1122/article/details/42048599
