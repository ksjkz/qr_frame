

from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import pandas as pd
'''
这个类,用于将pandas DataFrame转换为pyecharts图表
原生的pyecharts图表,需要手动设置x轴,y轴,以及groupby的列
而且输入数据格式只能是原生Python数据格式
借由这个类,,可以实现类似seaborn直接读取pandas DataFrame,然后直接生成图表的效果
示例:
plotter = pye_plot(df, 'month', 'ic', grouped_columns='class_name')
plotter.line_plot(title='IC',x_name='月份',y_name='IC',save_opt=True)
'''

class pye_plot():
       def __init__(self,df,x,y,grouped_columns):
              
               # 检查df是否为pandas DataFrame
              if not isinstance(df, pd.DataFrame):
                   raise ValueError("df必须是一个pandas DataFrame")
              if isinstance(grouped_columns, str):
                    grouped_columns = [grouped_columns]
              # 确保groupby_columns是列表或元组
              elif not isinstance(grouped_columns, (list, tuple)):
                     raise ValueError("groupby_columns必须是字符串、列表或元组")
              for column in grouped_columns:
                   if column not in df.columns:
                         raise ValueError(f"列 {column} 不存在于DataFrame中")
             
              if not isinstance(x, str):
                    raise ValueError("X必须是一个str")
              if not isinstance(y, str):
                    raise ValueError("y必须是一个str")
              self.df = df[[x,y, *grouped_columns]]
              self.grouped_columns = grouped_columns
              self.grouped_df = self.df.groupby(grouped_columns)
              self.df[x]= self.df[x].astype(str)
              self.x=list(self.df[x].astype(str).unique())
              self.y=y

       def line_plot(self,title='折线表',x_name='分组',y_name='数值',save_opt=False):
                line = (
                            Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
                            .add_xaxis(self.x)
                            .set_global_opts(
                    
                             title_opts=opts.TitleOpts(title=title),
                             xaxis_opts=opts.AxisOpts(name=x_name,axislabel_opts=opts.LabelOpts(rotate=45),splitline_opts = opts.SplitLineOpts(is_show=True)),
                             yaxis_opts=opts.AxisOpts(name=y_name,axislabel_opts=opts.LabelOpts(rotate=0),splitline_opts = opts.SplitLineOpts(is_show=True)),
                             legend_opts=opts.LegendOpts(pos_top="top")
                                                                                         
                        ))
                for name, group in self.grouped_df:
                     yy=group[self.y].to_list()
                     line.add_yaxis(name[0], yy, label_opts=opts.LabelOpts(is_show=False))



                if save_opt==True:
                           line.render("line.html")
                return line.render_notebook()  # 在 Jupyter Notebook 中使用
       
       def bar_plot(self,title='柱状图',x_name='分组',y_name='数值',save_opt=False):
                bar = (
                            Bar()
                            .add_xaxis(self.x)
                            .set_global_opts(
                             title_opts=opts.TitleOpts(title=title),
                             xaxis_opts=opts.AxisOpts(name=x_name,axislabel_opts=opts.LabelOpts(rotate=45)),
                             yaxis_opts=opts.AxisOpts(name=y_name,),
                             legend_opts=opts.LegendOpts(pos_top="top")
                             )
                        )
                for name, group in self.grouped_df:
                     yy=group[self.y].to_list()
                     bar.add_yaxis(name[0], yy, label_opts=opts.LabelOpts(is_show=False))
                if save_opt==True:
                           bar.render("bar.html")
                return bar.render_notebook()  # 在 Jupyter Notebook 中使用
