from bokeh.plotting import figure, save, show
from bokeh.models import Range1d, HoverTool
from bokeh.io import export_png
from operator import index
from posixpath import dirname
import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import os
import evaluateann
import datetime as dt
from panel.widgets import FloatSlider as fs
import datetime as dt
from trainann import output_locations

# Some hard-coded stuff for now - will move to a YAML config file
dir = os.path.dirname(os.path.realpath(__file__))
inp_template = os.path.join(dir,'ann_inp.csv')
dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['date'])
dfinps_global = dfinps.copy()
start_date = dt.datetime(2014, 1, 1)
end_date = dt.datetime(2014, 12, 31)
scale_df1 =pd.read_csv(os.path.join(dir,'input_scale.csv'),
                       index_col=0, parse_dates = ['month'])
scale_df = scale_df1.copy()

class SliderGroup:
    def __init__(self,input_loc):
        sp = dict(start=0,  end=2, step=0.1, value=1,
                  orientation = 'vertical',direction ='rtl',
                  margin=8,height=150)
        self.input_loc = input_loc
        self.fs1 = fs(name='Jan', **sp)
        self.fs2 = fs(name='Feb', **sp)
        self.fs3 = fs(name='Mar', **sp)
        self.fs4 = fs(name='Apr', **sp)
        self.fs5 = fs(name='May', **sp)
        self.fs6 = fs(name='Jun', **sp)
        self.fs7 = fs(name='Jul', **sp)
        self.fs8 = fs(name='Aug', **sp)
        self.fs9 = fs(name='Sep', **sp)
        self.fs10 = fs(name='Oct', **sp)
        self.fs11 = fs(name='Nov', **sp)
        self.fs12 = fs(name='Dec', **sp)

        self.fs_set=[self.fs1,self.fs2,self.fs3,self.fs4,self.fs5,self.fs6,
                    self.fs7,self.fs8,self.fs9,self.fs10,self.fs11,self.fs12]

        self.kwargs = dict(fs1=self.fs1,fs2=self.fs2,fs3=self.fs3,fs4=self.fs4,
                 fs5=self.fs5,fs6=self.fs6,fs7=self.fs7,fs8=self.fs8,
                 fs9=self.fs9,fs10=self.fs10,fs11=self.fs11,fs12=self.fs12)

def scale_inputs(inp_template,input_loc,scale_df,fs1,fs2,fs3,
                 fs4,fs5,fs6,fs7,fs8,fs9,fs10,fs11,fs12):
                 
    global dfinps_global
    dfinps = pd.read_csv(inp_template,index_col=0, parse_dates = ['date'])

    scale_df.loc[1,input_loc] = fs1
    scale_df.loc[2,input_loc] = fs2
    scale_df.loc[3,input_loc] = fs3
    scale_df.loc[4,input_loc] = fs4
    scale_df.loc[5,input_loc] = fs5
    scale_df.loc[6,input_loc] = fs6
    scale_df.loc[7,input_loc] = fs7
    scale_df.loc[8,input_loc] = fs8
    scale_df.loc[9,input_loc] = fs9
    scale_df.loc[10,input_loc] = fs10
    scale_df.loc[11,input_loc] = fs11
    scale_df.loc[12,input_loc] = fs12

    for mon in scale_df.index:
        dfmod = dfinps.loc[dfinps.index.month == mon ,input_loc]*scale_df.loc[mon,input_loc]
        dfinps_global.update(dfmod, overwrite=True)
    return dfinps_global

def make_input_plot(dfinp,input_loc,start_date,end_date):
    #print(dfinp.head())
    p = figure(title = "",x_axis_type='datetime')
    p.line(source = dfinp,x='date',y=str(input_loc), line_color = 'blue',
           line_dash = 'solid', line_width=1, legend_label=input_loc)
    p.plot_height = 350
    p.plot_width = 700
    p.x_range = Range1d(start=start_date, end=end_date)
    return p

def make_ts_plot_ANN(selected_key_stations,dfinp,start_date,end_date,
                     refresh,listener):
    refresh = refresh
    listener = listener
    targ_df,pred_df = evaluateann.run_ann(selected_key_stations,dfinp)
    print(pred_df.head())
    p = figure(title = "",x_axis_type='datetime')
    p.line(source = targ_df,x='index',y=str(selected_key_stations),
           line_color = 'red', line_width=1, legend_label='Historical')
    p.line(source = pred_df,x='date',y=str(selected_key_stations),
           line_color = 'blue', line_width=1, legend_label='Predicted')
    p.plot_height = 500
    p.plot_width = 1200
    p.legend.location = 'top_left'
    p.yaxis.axis_label = 'EC (uS/cm)'
    p.xaxis.axis_label = 'Date'
    p.x_range = Range1d(start=start_date, end=end_date)

    tt = [
    ("Value:", "$y{0,0.0}"),
    ("Date:", "$x{%F}"),
    ]

    p.add_tools(HoverTool(
        tooltips = tt,
        formatters = {'$x':'datetime'}
    ))

    return p

def listener(e1,e2,e3,e4,e5,e6):
    e1 = e1
    e2 = e2
    e3 = e3
    e4 = e4
    e5 = e5
    e6 = e6
    return None

# Widgets

inputlocs = ['northern_flow','exports']
inputlocs_w = pn.widgets.Select(name='Input Location', options = inputlocs,
                                value = 'northern_flow')
variables = output_locations
variables_w = pn.widgets.Select(name='Output Location', options = variables)
architecture_w = pn.widgets.Select(name='ML Model', options = ['MLP', 'LSTM', 'GRU', 'ResNet'])
dateselect_w = pn.widgets.DateRangeSlider(name='Date Range Slider',
                                            start=dt.datetime(1990, 1, 1),
                                            end=dt.datetime(2019, 12, 31),
                                            value=(start_date, end_date),
                                            disabled =True)
run_btn = pn.widgets.Button(name='Run ANN', button_type='primary')
train_btn = pn.widgets.Button(name='Train ANN', button_type='primary')
refresh_btn = pn.widgets.Button(name='Refresh Plot', button_type='default',width=50)

title_pane = pn.pane.Markdown('''
### DSM2 Emulator Dashboard
''')
assumptions_pane = pn.pane.Markdown('''
#### Hyperparameters
MLP Network  
model_str_def = 'd8_d4_o1'  
input_prepro_option=1  
group_stations = False  
epochs = 5000  
input file = dsm2_ann_inputs_20220204.xlsx
''')

# Bindings

northern_flow = SliderGroup('northern_flow')
scale_northern_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = northern_flow.input_loc,inp_template = inp_template,
                           **northern_flow.kwargs)

exports = SliderGroup('exports')
scale_exp = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = exports.input_loc,inp_template = inp_template,
                           **exports.kwargs)

sjr_flow = SliderGroup('sjr_flow')
scale_sjr_flow = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_flow.input_loc,inp_template = inp_template,
                           **sjr_flow.kwargs)

net_delta_cu = SliderGroup('net_delta_cu')
scale_net_delta_cu = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = net_delta_cu.input_loc,inp_template = inp_template,
                           **net_delta_cu.kwargs)

sjr_vernalis_ec = SliderGroup('sjr_vernalis_ec')
scale_sjr_vernalis_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sjr_vernalis_ec.input_loc,inp_template = inp_template,
                           **sjr_vernalis_ec.kwargs)

sac_greens_ec = SliderGroup('sac_greens_ec')
scale_sac_greens_ec = pn.bind(scale_inputs,scale_df = scale_df,
                           input_loc = sac_greens_ec.input_loc,inp_template = inp_template,
                           **sac_greens_ec.kwargs)


listener_bnd = pn.bind(listener,
                       e1 = scale_northern_flow,
                       e2 = scale_exp,
                       e3 = scale_sjr_flow,
                       e4 = scale_net_delta_cu,
                       e5 = scale_sjr_vernalis_ec,
                       e6 = scale_sac_greens_ec)

# Dashboard Layout

dash = pn.Row(
    pn.Column(pn.pane.Markdown('### ANN Inputs - Input Scaler'),
            
            pn.Tabs(
                ("Northern Flow",
                pn.Column(
                pn.Row(*northern_flow.fs_set),
                pn.bind(make_input_plot,dfinp=scale_northern_flow,input_loc='northern_flow',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("Exports",
                pn.Column(
                pn.Row(*exports.fs_set),
                pn.bind(make_input_plot,dfinp=scale_exp,input_loc='exports',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("SJR flow",
                pn.Column(
                pn.Row(*sjr_flow.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sjr_flow,input_loc='sjr_flow',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("Net Delta Consumptive Use",
                pn.Column(
                pn.Row(*net_delta_cu.fs_set),
                pn.bind(make_input_plot,dfinp=scale_net_delta_cu,input_loc='net_delta_cu',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("SJR Vernalis EC",
                pn.Column(
                pn.Row(*sjr_vernalis_ec.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sjr_vernalis_ec,input_loc='sjr_vernalis_ec',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("Sac Greens EC",
                pn.Column(
                pn.Row(*sac_greens_ec.fs_set),
                pn.bind(make_input_plot,dfinp=scale_sac_greens_ec,input_loc='sac_greens_ec',
                    start_date=dateselect_w.value[0],end_date=dateselect_w.value[1]))),

                ("DXC",
                pn.Column()),
            )
    ),

    pn.Column(pn.pane.Markdown('### ANN Outputs'),variables_w,dateselect_w,
              pn.bind(make_ts_plot_ANN,
                      selected_key_stations=variables_w,
                      dfinp = dfinps_global,
                      start_date=dateselect_w.value[0],
                      end_date=dateselect_w.value[1],
                      refresh=refresh_btn, 
                      listener = listener_bnd
              ),
              architecture_w,refresh_btn
    )
)

dash.show(title = "DSM2 ANN Emulator Dashboard")