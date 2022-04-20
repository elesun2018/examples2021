# -*- coding: utf-8 -*-
"""
   create-pdf-03.py
fun:
    创建pdf文档
env:
    reportlab-3.5.68
ref:
    https://zhuanlan.zhihu.com/p/318390273
"""
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("out03.pdf")
styles = getSampleStyleSheet()
style = styles['Normal']
story =[]

t = Image(".\\img.jpg")
story.append(t)

doc.build(story)