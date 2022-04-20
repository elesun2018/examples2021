# -*- coding: utf-8 -*-
"""
   create-pdf-01.py
fun:
    创建pdf文档
env:
    reportlab-3.5.68
ref:
    https://zhuanlan.zhihu.com/p/318390273
"""
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# 调用模板，创建指定名称的PDF文档
doc = SimpleDocTemplate("out01.pdf")
# 获得模板表格
styles = getSampleStyleSheet()
# 指定模板
style = styles['Normal']
# 初始化内容
story =[]
# 将段落添加到内容中
story.append(Paragraph("This is the first Document!",style))
# 将内容输出到PDF中
doc.build(story)