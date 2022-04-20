# -*- coding: utf-8 -*-
"""
   create-pdf-04.py
fun:
    创建pdf文档
env:
    reportlab-3.5.68
ref:
    https://zhuanlan.zhihu.com/p/318390273
"""
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image
from reportlab.lib.styles import getSampleStyleSheet

# 调用模板，创建指定名称的PDF文档
doc = SimpleDocTemplate("out04.pdf")
# 指定模板
styles = getSampleStyleSheet()
style = styles['Normal']
story =[]

# 创建
str1 = Paragraph("This is a table following !",style)
data= [['00', '01', '02', '03', '04'],
       ['10', '11', '12', '13', '14'],
       ['20', '21', '22', '23', '24'],
       ['30', '31', '32', '33', '34']]
tab = Table(data)
str2 = Paragraph("This is a image following !",style)
img = Image(".\\img.jpg")

# 添加
story.append(str1)
story.append(tab)
story.append(str2)
story.append(img)
doc.build(story)